"""Versioned response/preference JSONL and loss masks shared by phi trainers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .hf_backend import PromptCodec


def write_records(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({**row, "schema_version": 3}, ensure_ascii=False)+"\n")


def read_records(path: str | Path, mode: str) -> list[dict]:
    if mode not in {"sft", "dpo"}:
        raise ValueError("Expected sft or dpo")
    rows, inputs, signatures = [], {}, set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.pop("schema_version", None) != 3 or type(row.get("synthetic")) is not bool:
                raise ValueError(f"Line {number}: version and explicit synthetic provenance required")
            fields = ("prompt", "response") if mode == "sft" else ("prompt", "chosen", "rejected")
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
                raise ValueError(f"Line {number}: nonempty {fields} required")
            input_id = row.get("input_id") or hashlib.sha256(row["prompt"].encode()).hexdigest()
            if input_id in inputs and inputs[input_id] != row["prompt"]:
                raise ValueError("The same input ID identifies conflicting prompts")
            inputs[input_id] = row["prompt"]
            signatures.add((row["synthetic"], row.get("snapshot_id"), row.get("deployment_id"),
                            row.get("cost_unit"), row.get("reference_cost")))
            rows.append(row)
    if not rows:
        raise ValueError("Empty Specialist data")
    if len(signatures) != 1:
        raise ValueError("Mixed synthetic provenance, deployments or receiver snapshots")
    return rows


def response_batch(codec: PromptCodec, pairs: list[tuple[str, str]], max_length: int = 8192,
                   device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    if not pairs:
        raise ValueError("Empty response batch")
    tokenizer = codec.tokenizer
    sequences, labels = [], []
    for prompt, response in pairs:
        if not response.strip():
            raise ValueError("Empty response cannot train phi")
        prefix = codec.input_ids(prompt)
        suffix = tokenizer.encode(response, add_special_tokens=False)
        eos = tokenizer.eos_token_id
        if eos is not None and (not suffix or suffix[-1] != eos):
            suffix.append(eos)
        if not prefix or not suffix or len(prefix)+len(suffix) > max_length:
            raise ValueError("Training length limit exceeded; refusing silent truncation")
        sequences.append(prefix+suffix)
        labels.append([-100]*len(prefix)+suffix)
    length = max(map(len, sequences))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires a pad or EOS token")
    return {
        "input_ids": torch.tensor([s+[pad]*(length-len(s)) for s in sequences], device=device),
        "attention_mask": torch.tensor([[1]*len(s)+[0]*(length-len(s)) for s in sequences], device=device),
        "labels": torch.tensor([s+[-100]*(length-len(s)) for s in labels], device=device),
    }
