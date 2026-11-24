"""Executable PEFT SFT / fixed-reference DPO+SFT trainer for Specialist phi.

Real weights are loaded only when the CLI/file runner is explicitly invoked.
One process owns one model (optionally QLoRA); this is not a DDP/FSDP trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from .hf_backend import PromptCodec, _input_device
from .identity import artifact_digest
from .schema import SpecialistRecord
from .specialist_data import read_records, response_batch
from .training import dpo_loss, sequence_log_probs, sft_loss


def _forward(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits


def reference_scores(model, codec, records, max_length: int = 8192) -> list[tuple[float, float]]:
    model.eval()
    device = _input_device(model)
    result = []
    with torch.no_grad():
        for row in records:
            values = []
            for name in ("chosen", "rejected"):
                batch = response_batch(codec, [(row["prompt"], row[name])], max_length, device)
                values.append(float(sequence_log_probs(_forward(model, batch), batch["labels"])[0]))
            result.append(tuple(values))
    return result


def optimize_specialist(model, codec: PromptCodec, records: list[dict], mode: str,
                        steps: int = 100, batch_size: int = 1, accumulation: int = 1,
                        lr: float = 2e-5, beta: float = .1, aux_weight: float = .1,
                        max_length: int = 8192, seed: int = 7, reference_model=None,
                        ref_logps: list[tuple[float, float]] | None = None,
                        auxiliary_records: list[dict] | None = None) -> dict:
    if mode not in {"sft", "dpo"} or not records:
        raise ValueError("Nonempty records and sft/dpo mode required")
    if min(steps, batch_size, accumulation, max_length) <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Positive training settings required")
    if not math.isfinite(aux_weight) or aux_weight < 0:
        raise ValueError("Auxiliary weight must be nonnegative")
    reference_seconds = 0.
    if mode == "dpo":
        if reference_model is model:
            raise ValueError("The policy cannot also be its mutable reference")
        if ref_logps is None:
            if reference_model is None:
                raise ValueError("DPO requires a frozen reference or precomputed reference log probabilities")
            reference_model.requires_grad_(False)
            start = time.perf_counter()
            ref_logps = reference_scores(reference_model, codec, records, max_length)
            reference_seconds = time.perf_counter()-start
        if len(ref_logps) != len(records) or any(not all(math.isfinite(x) for x in r) for r in ref_logps):
            raise ValueError("Reference scores do not match the preference data")
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters() if not parameter.requires_grad}
    if not trainable:
        raise ValueError("No trainable adapter parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    rng = random.Random(seed)
    device = _input_device(model)

    def batch_for(indices, field):
        return response_batch(codec, [(records[i]["prompt"], records[i][field]) for i in indices], max_length, device)

    def margin():
        if mode != "dpo":
            return None
        policy = reference_scores(model, codec, records, max_length)
        return sum((a-b)-(r[0]-r[1]) for (a, b), r in zip(policy, ref_logps))/len(records)

    initial_margin = margin()
    losses, gradient_norms, update_norms = [], [], []
    start = time.perf_counter()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(accumulation):
            indices = [rng.randrange(len(records)) for _ in range(batch_size)]
            if mode == "sft":
                batch = batch_for(indices, "response")
                loss = sft_loss(_forward(model, batch), batch["labels"])
            else:
                chosen, rejected = batch_for(indices, "chosen"), batch_for(indices, "rejected")
                chosen_logits = _forward(model, chosen)
                log_chosen = sequence_log_probs(chosen_logits, chosen["labels"])
                log_rejected = sequence_log_probs(_forward(model, rejected), rejected["labels"])
                fixed = torch.tensor([ref_logps[i] for i in indices], device=log_chosen.device)
                loss = dpo_loss(log_chosen, log_rejected, fixed[:, 0], fixed[:, 1], beta)
                if aux_weight:
                    if auxiliary_records:
                        rows = [rng.choice(auxiliary_records) for _ in range(batch_size)]
                        aux = response_batch(codec, [(r["prompt"], r["response"]) for r in rows], max_length, device)
                        loss = loss+aux_weight*sft_loss(_forward(model, aux), aux["labels"])
                    else:
                        loss = loss+aux_weight*sft_loss(chosen_logits, chosen["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Specialist loss")
            (loss/accumulation).backward()
            total += float(loss.detach())/accumulation
        gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(trainable, 1.)))
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer.step()
        update_norms.append(sum(float((parameter.detach()-old).float().square().sum()) for parameter, old in zip(trainable, before))**.5)
        losses.append(total)
    seconds = time.perf_counter()-start
    model.eval()
    frozen_unchanged = all(parameter._version == frozen_versions[name] for name, parameter in model.named_parameters() if name in frozen_versions)
    if not frozen_unchanged:
        raise RuntimeError("Frozen model or reference parameters changed during optimization")
    return {"mode": mode, "steps": steps, "examples": len(records), "initial_loss": losses[0],
            "final_loss": losses[-1], "initial_margin": initial_margin, "final_margin": margin(),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "per_step_loss": losses, "gradient_norms": gradient_norms, "parameter_update_norms": update_norms, "learning_rate": lr,
            "frozen_parameters_unchanged": frozen_unchanged,
            "optimization_seconds": seconds, "reference_seconds": reference_seconds}


def train_from_file(data: str | Path, output: str | Path, model_spec: dict, mode: str = "sft",
                    init_adapter: str | None = None, reference_adapter: str | None = None,
                    aux_data: str | Path | None = None, lora_rank: int = 16, lora_alpha: float = 32.,
                    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
                    load_in_4bit: bool = False, gradient_checkpointing: bool = True, **settings) -> dict:
    rows = read_records(data, mode)
    aux = read_records(aux_data, "sft") if aux_data else None
    for row in rows:
        SpecialistRecord.parse(row["response"] if mode == "sft" else row["chosen"])
    if aux:
        if aux[0]["synthetic"] != rows[0]["synthetic"]:
            raise ValueError("Auxiliary and primary data have inconsistent synthetic provenance")
        for row in aux:
            SpecialistRecord.parse(row["response"])
    if mode == "dpo" and (not reference_adapter or not init_adapter):
        raise ValueError("DPO requires explicit initial-policy and fixed SFT-reference adapter paths")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError('Install optional training dependencies: pip install -e ".[models]"') from exc
    torch.manual_seed(settings.get("seed", 7))
    spec = dict(model_spec)
    name = spec["name_or_path"]
    device = spec.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[spec.get("dtype", "bfloat16")]
    if dtype == torch.float16:
        raise ValueError("Use bfloat16 or float32: this trainer does not implement fp16 loss scaling")
    tokenizer = AutoTokenizer.from_pretrained(spec.get("tokenizer", name), use_fast=True,
                                             revision=spec.get("revision"), trust_remote_code=False)
    codec = PromptCodec(tokenizer, spec.get("chat_template_kwargs", {}), spec.get("chat", True))
    kwargs = {"torch_dtype": dtype, "revision": spec.get("revision"), "trust_remote_code": False}
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("The provided QLoRA loader requires CUDA and bitsandbytes")
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
            device_map={"": device})
    base = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not load_in_4bit:
        base.to(device)
    if load_in_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    base.config.use_cache = False
    ref_logps, ref_seconds = None, 0.
    if mode == "dpo":
        # A small reference adapter shares the frozen base. Precompute fixed
        # scores before loading/training the policy adapter; no second base LM.
        model = PeftModel.from_pretrained(base, reference_adapter, adapter_name="reference", is_trainable=False)
        model.requires_grad_(False)
        start = time.perf_counter()
        ref_logps = reference_scores(model, codec, rows, settings.get("max_length", 8192))
        ref_seconds = time.perf_counter()-start
        model.load_adapter(init_adapter, adapter_name="default", is_trainable=True)
        model.set_adapter("default")
    elif init_adapter:
        model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=lora_rank, lora_alpha=lora_alpha,
            target_modules=list(lora_targets), lora_dropout=0., bias="none", task_type="CAUSAL_LM"))
    # Keep stochastic dropout out of preference/reference comparisons.
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any("lora_" not in n or ".default." not in n for n in trainable_names):
        raise ValueError("Only the policy's default LoRA adapter may be trainable")
    metrics = optimize_specialist(model, codec, rows, mode, ref_logps=ref_logps,
                                  auxiliary_records=aux, **settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, selected_adapters=["default"])
    tokenizer.save_pretrained(output)
    metrics.update(synthetic=rows[0]["synthetic"], model=name, reference_adapter=reference_adapter,
                   reference_adapter_sha256=artifact_digest(reference_adapter) if reference_adapter else None,
                   initialized_adapter_sha256=artifact_digest(init_adapter) if init_adapter else None,
                   reference_seconds=ref_seconds, adapter=str(output),
                   data_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest())
    (output/"training_manifest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True, help="JSON model spec, identical to backends.hf models[id]")
    parser.add_argument("--mode", choices=("sft", "dpo"), default="sft")
    parser.add_argument("--init-adapter")
    parser.add_argument("--reference-adapter")
    parser.add_argument("--aux-data", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--aux-weight", type=float, default=.1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = vars(parser.parse_args())
    spec = json.loads(args.pop("model_config").read_text(encoding="utf-8"))
    args["gradient_checkpointing"] = not args.pop("no_gradient_checkpointing")
    print(json.dumps(train_from_file(model_spec=spec, **args), indent=2))


if __name__ == "__main__":
    main()
