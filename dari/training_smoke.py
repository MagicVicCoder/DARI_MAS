"""Recheck real Specialist optimization and adapter reload without utility claims."""
import gc
import argparse
import json
from pathlib import Path

import torch

from .backends import GenerationRequest
from .hf_backend import HuggingFaceBackend
from .train_specialist import train_from_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "dpo", "reload"), default="all")
    parser.add_argument("--device", default="cuda:4")
    args = parser.parse_args()
    root = Path("outputs/qwen3/validation")
    output = Path("outputs/qwen3/training_checks")
    config = json.loads(Path("configs/qwen3_8b_debug.json").read_text())
    spec = config["backend"]["models"]["A"]
    spec["device"] = args.device
    spec["chat_template_kwargs"] = {"enable_thinking": False}
    if args.stage == "all":
        warm = train_from_file(root/"sft.jsonl", output/"sft_adapter", spec,
            steps=1, lora_rank=2, lora_alpha=4, max_length=4096, lr=2e-5)
    else:
        warm = json.loads((output/"sft_adapter/training_manifest.json").read_text())
    gc.collect()
    torch.cuda.empty_cache()
    if args.stage != "reload":
        trained = train_from_file(root/"dpo_smoke.jsonl", output/"dpo_adapter", spec, mode="dpo",
            init_adapter=str(output/"sft_adapter"), reference_adapter=str(output/"sft_adapter"),
            steps=1, max_length=4096, lr=2e-5)
    else:
        trained = json.loads((output/"dpo_adapter/training_manifest.json").read_text())
    gc.collect()
    torch.cuda.empty_cache()
    backend = HuggingFaceBackend.from_specs({"S": {**spec, "adapter": str(output/"dpo_adapter")}})
    result = backend.generate(GenerationRequest("S", "specialist", "Return a brief greeting.", 16, 0))
    (output/"summary.json").write_text(json.dumps({"sft": warm, "dpo": trained,
        "reload": {"output": result.text, "output_tokens": result.usage.output_tokens,
                   "hidden_dimension": len(result.features["output"].values),
                   "adapter_id": backend.deployment_identity["adapters"]["S"]}}, indent=2))


if __name__ == "__main__":
    main()
