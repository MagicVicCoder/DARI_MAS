"""Receiver-driven theta/phi alternation with data refresh after each phi update."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path

import torch

from .configuration import load_dataset
from .train_controller import train_from_targets
from .train_specialist import train_from_file
from .workflows import artifact_digest, collect_phase


def _release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_alternating(config: dict, output: str | Path, specialist_trainer=train_from_file) -> dict:
    config = deepcopy(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    rounds = training.get("rounds", 2)
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("Positive number of alternating rounds required")
    tasks = load_dataset(training["tasks"], require_gold=True)
    phi_settings = dict(training["specialist"])
    model_spec = phi_settings.pop("model_spec")
    theta_settings = dict(training.get("controller", {}))
    adapter = training.get("initial_adapter")
    if adapter:
        reference = training.get("reference_adapter", adapter)
    else:
        if not training.get("sft_data"):
            raise ValueError("Provide protocol SFT data or an existing SFT adapter")
        warm = specialist_trainer(training["sft_data"], output/"sft_adapter", model_spec, mode="sft", **phi_settings)
        adapter = warm["adapter"]
        reference = adapter
        (output/"warmup.json").write_text(json.dumps(warm, indent=2), encoding="utf-8")
    initial_phi = artifact_digest(adapter)
    controller = config.get("controller_checkpoint")
    history = []
    for index in range(rounds):
        root = output/f"round_{index:02d}"
        root.mkdir(parents=True, exist_ok=True)
        controller_data = collect_phase(config, tasks, root/"theta_data", "controller", controller, adapter,
                                        allow_specialist_update=True)
        _release_models()
        theta = train_from_targets(root/"theta_data"/"targets.jsonl", root/"controller.pt",
                                   resume=controller, allow_specialist_update=True, **theta_settings)
        controller = theta["checkpoint"]
        specialist_data = collect_phase(config, tasks, root/"phi_data", "specialist", controller, adapter)
        _release_models()
        if specialist_data["preference_pairs"] == 0:
            raise ValueError(f"Round {index}: no reliable receiver preference pairs. Data saved in {root / 'phi_data'}; "
                             "increase coverage/repeats or inspect the receiver before training phi")
        phi = specialist_trainer(root/"phi_data"/"preferences.jsonl", root/"specialist_adapter", model_spec,
            mode="dpo", init_adapter=adapter, reference_adapter=reference,
            aux_data=root/"phi_data"/"auxiliary.jsonl" if specialist_data["auxiliary_examples"] else None,
            **phi_settings)
        adapter = phi["adapter"]
        _release_models()
        history.append({"round": index, "theta": theta, "phi": phi,
                        "theta_collection": controller_data, "phi_collection": specialist_data})
        (root/"round_manifest.json").write_text(json.dumps(history[-1], indent=2), encoding="utf-8")
    # Phi just changed. Refresh theta's policy-evaluation data before exporting
    # the final pair, instead of deploying theta trained against the previous phi.
    refreshed = collect_phase(config, tasks, output/"final_theta_data", "controller", controller, adapter,
                              allow_specialist_update=True)
    _release_models()
    final = train_from_targets(output/"final_theta_data"/"targets.jsonl", output/"controller.pt",
                               resume=controller, allow_specialist_update=True, **theta_settings)
    config["controller_checkpoint"] = final["checkpoint"]
    config.setdefault("model_adapters", {})[config.get("runtime", {}).get("specialist_model", "S")] = adapter
    (output/"deployment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"rounds_completed": len(history), "controller": final["checkpoint"],
        "specialist_adapter": adapter, "reference_adapter": reference,
        "final_controller_adapter": adapter, "initial_phi_snapshot": initial_phi,
        "final_phi_snapshot": artifact_digest(adapter), "synthetic": refreshed["synthetic"],
        "final_theta_collection": refreshed, "history": history}
    (output/"training_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    print(json.dumps(run_alternating(json.loads(args.config.read_text(encoding="utf-8")), args.output), indent=2))


if __name__ == "__main__":
    main()
