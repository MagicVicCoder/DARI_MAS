"""Small, download-free models/tokenizers for exercising our model boundary."""
from types import SimpleNamespace

import torch
from torch import nn


class CharacterTokenizer:
    is_fast = True
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join("<" + m["role"] + ">" + m["content"] for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 3 for c in text]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(chr(int(i)-3) for i in ids if int(i) > 2)

    def __call__(self, text, return_offsets_mapping=False, return_tensors=None,
                 add_special_tokens=False, **kwargs):
        result = {"input_ids": self.encode(text), "attention_mask": [1]*len(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result


class DeterministicCachedLM(nn.Module):
    """A tiny tensor forward with the same KV/hidden contract as a causal LM."""
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=4096)
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=True, output_hidden_states=True, **kwargs):
        self.calls += 1
        phase = past_key_values[0] if past_key_values else 0
        token = [ord("O")+3, ord("K")+3, 2][min(phase, 2)]
        logits = torch.full((*input_ids.shape, 259), -1000., device=input_ids.device)
        logits[..., token] = 0.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :, None]*1000
        hidden = input_ids.float().unsqueeze(-1) + positions + torch.arange(4, device=input_ids.device)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=(phase+1,))


class TrainableCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(259, 8)
        self.q_proj = nn.Linear(8, 8)
        self.head = nn.Linear(8, 259)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids).cumsum(1)
        return SimpleNamespace(logits=self.head(torch.tanh(self.q_proj(hidden))))
