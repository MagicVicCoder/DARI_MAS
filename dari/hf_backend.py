"""Local causal LM adapter with exact chat inputs and single-pass hidden pooling.

Transformers/PEFT are optional and imported only by ``from_specs``. Generation is
serial, one sequence per call, with normal within-call KV caching. There is no
cross-call KV state, independent feature encoder, or automatic prompt truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import time
from typing import Any

import torch

from .backends import Generation, GenerationRequest
from .costs import Usage
from .identity import artifact_digest, fingerprint
from .schema import Feature


class PromptCodec:
    """Shared input formatting for inference, cost estimates and response training."""
    def __init__(self, tokenizer, chat_template_kwargs: dict | None = None, chat: bool = True):
        self.tokenizer, self.chat = tokenizer, chat
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def format(self, prompt: str) -> str:
        if not self.chat:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, **self.chat_template_kwargs)

    def input_ids(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(self.format(prompt), add_special_tokens=False)

    def count_input(self, prompt: str) -> int:
        return len(self.input_ids(prompt))

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


@dataclass
class ModelHandle:
    model: Any
    codec: PromptCodec
    context_limit: int = 32768
    temperature: float = 0.
    top_p: float = 1.
    identity: dict | None = None
    adapter_id: str = "base"

    def __post_init__(self):
        if self.context_limit <= 0 or self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid context/sampling configuration")


def _input_device(model):
    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    return embedding.weight.device if embedding is not None else next(model.parameters()).device


class HuggingFaceBackend:
    synthetic = False

    def __init__(self, handles: dict[str, ModelHandle]):
        if not handles:
            raise ValueError("At least one model handle required")
        if len({id(h.model) for h in handles.values()}) != len(handles):
            raise ValueError("The same model object cannot be shared across mutable role adapters")
        self.handles = dict(handles)
        self.last_latency_ms = 0.
        for h in self.handles.values():
            if not isinstance(h.identity, dict) or not h.identity:
                raise ValueError("Manually loaded models require explicit frozen model/tokenizer identity")
            h.model.eval()
            h.model.requires_grad_(False)
        self.deployment_identity = {
            "engine": "hf-cached-forward-v2", "representation": "final-hidden-named-ranges-v2",
            "models": {key: {"identity": h.identity, "context_limit": h.context_limit,
                             "temperature": h.temperature, "top_p": h.top_p} for key, h in self.handles.items()},
            "adapters": {key: h.adapter_id for key, h in self.handles.items()}}

    @classmethod
    def from_specs(cls, specs: dict[str, dict]) -> HuggingFaceBackend:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('Install the optional model dependencies: pip install -e ".[models]"') from exc
        handles = {}
        for model_id, original in specs.items():
            spec = dict(original)
            path = spec.pop("name_or_path")
            tokenizer_path = spec.pop("tokenizer", path)
            adapter = spec.pop("adapter", None)
            revision = spec.pop("revision", None)
            declared_revision = spec.pop("deployment_revision", None)
            dtype_name = spec.pop("dtype", "bfloat16")
            dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
            model_kwargs = dict(spec.pop("model_kwargs", {}))
            if model_kwargs.get("trust_remote_code"):
                raise ValueError("Remote code execution is disabled; load a vetted model in a custom factory")
            device_map = spec.pop("device_map", None)
            device = spec.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=revision,
                                                      use_fast=True, trust_remote_code=False)
            if not tokenizer.is_fast:
                raise ValueError("A fast tokenizer with character offsets is required for span pooling")
            kwargs = {"torch_dtype": dtype, "revision": revision, "trust_remote_code": False, **model_kwargs}
            if device_map is not None:
                kwargs["device_map"] = device_map
            model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
            resolved_revision = getattr(model.config, "_commit_hash", None) or declared_revision or revision
            if not resolved_revision:
                raise ValueError("Local base weights require deployment_revision (an immutable artifact identity)")
            if device_map is None:
                model.to(device)
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
            codec = PromptCodec(tokenizer, spec.pop("chat_template_kwargs", {}), spec.pop("chat", True))
            identity = {"model": str(path), "revision": resolved_revision, "dtype": dtype_name,
                        "tokenizer": fingerprint(tokenizer.backend_tokenizer.to_str()),
                        "chat_template": fingerprint(tokenizer.chat_template), "chat": codec.chat,
                        "chat_template_kwargs": codec.chat_template_kwargs, "model_kwargs": model_kwargs}
            allowed = {"context_limit", "temperature", "top_p"}
            if set(spec)-allowed:
                raise ValueError(f"Unknown model settings: {sorted(set(spec)-allowed)}")
            handles[model_id] = ModelHandle(model, codec, identity=identity,
                adapter_id=artifact_digest(adapter) if adapter else "base", **spec)
        return cls(handles)

    def count_tokens(self, model_id: str, text: str) -> int:
        return self.handles[model_id].codec.count_input(text)

    def count_text_tokens(self, model_id: str, text: str) -> int:
        return self.handles[model_id].codec.count_text(text)

    def generate(self, request: GenerationRequest) -> Generation:
        h = self.handles[request.model_id]
        if request.max_new_tokens <= 0:
            raise ValueError("Positive generation quota required")
        tokenizer = h.codec.tokenizer
        formatted = h.codec.format(request.prompt)
        encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if not ids or len(ids) + request.max_new_tokens > h.context_limit:
            raise ValueError("Requested prompt/output exceeds model context limit; no input was truncated")
        content_offset = formatted.find(request.prompt)
        if (content_offset < 0 or formatted.find(request.prompt, content_offset+1) >= 0) and request.feature_texts:
            raise ValueError("Chat template does not preserve one unambiguous user prompt for feature ranges")
        for key in request.feature_texts:
            if key not in request.feature_spans:
                raise ValueError(f"Missing ID-specific feature range: {key}")
            begin, end = request.feature_spans[key]
            if not 0 <= begin < end <= len(request.prompt):
                raise ValueError(f"Invalid feature range: {key}")
        device = _input_device(h.model)
        devices = sorted({p.device.index for p in h.model.parameters() if p.device.type == "cuda"})
        seed_text = f"{request.seed}:{request.role}:{request.call_index}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big") % (2**63-1)
        features, generated, output_hidden = {}, [], []
        past = None
        current = torch.tensor([ids], dtype=torch.long, device=device)
        eos = getattr(getattr(h.model, "generation_config", None), "eos_token_id", tokenizer.eos_token_id)
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        accepts_keep = "logits_to_keep" in inspect.signature(h.model.forward).parameters
        start_time = time.perf_counter()
        # Fork only the RNGs touched by this model. No persistent sampling state
        # needs to be copied with EpisodeState; seeded branch replay is explicit.
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(seed)
            for step in range(request.max_new_tokens):
                mask = torch.ones(1, len(ids)+step, dtype=torch.long, device=device)
                kwargs = dict(input_ids=current, attention_mask=mask, past_key_values=past,
                              use_cache=True, output_hidden_states=True, return_dict=True)
                if accepts_keep:
                    kwargs["logits_to_keep"] = 1
                out = h.model(**kwargs)
                hidden = out.hidden_states[-1][0].detach()
                if step == 0:
                    for key in request.feature_texts:
                        begin, end = request.feature_spans[key]
                        begin, end = begin+content_offset, end+content_offset
                        positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > begin]
                        if positions:
                            values = hidden[positions].float().mean(0).cpu().tolist()
                            features[key] = Feature(request.role, values)
                else:
                    output_hidden.append(hidden[-1].float())
                logits = out.logits[0, -1].float()
                if h.temperature == 0:
                    token = int(logits.argmax())
                else:
                    probabilities = torch.softmax(logits/h.temperature, dim=-1)
                    if h.top_p < 1:
                        probs, order = probabilities.sort(descending=True)
                        remove = probs.cumsum(0)-probs >= h.top_p
                        probs[remove] = 0
                        token = int(order[torch.multinomial(probs, 1)])
                    else:
                        token = int(torch.multinomial(probabilities, 1))
                generated.append(token)
                past = out.past_key_values
                if token in eos:
                    break
                current = torch.tensor([[token]], dtype=torch.long, device=device)
            # At the length cap the final emitted token has not been fed back.
            # Pool only already-computed representations; never run an extra LM
            # forward to encode the last token merely for the controller.
            vectors = output_hidden or [hidden[-1].float()]
            features["output"] = Feature(request.role, torch.stack(vectors).mean(0).cpu().tolist())
            text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            draft_start = text.find('"draft"')
            if draft_start >= 0 and output_hidden:
                colon = text.find(":", draft_start)
                start = colon+1
                while start < len(text) and text[start].isspace():
                    start += 1
                try:
                    _, length = json.JSONDecoder().raw_decode(text[start:])
                    end = start+length
                    selected, prior = [], 0
                    for i, vector in enumerate(output_hidden):
                        following = len(tokenizer.decode(generated[:i+1], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False))
                        if following > start and prior < end:
                            selected.append(vector)
                        prior = following
                    if selected:
                        features["draft"] = Feature(request.role, torch.stack(selected).mean(0).cpu().tolist())
                except ValueError:
                    pass
        self.last_latency_ms = (time.perf_counter()-start_time)*1000
        return Generation(text, Usage(request.model_id, len(ids), len(generated), context_tokens=len(ids)), features)
