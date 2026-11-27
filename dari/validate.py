"""Small real-model engineering validation, never a benchmark or utility experiment."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import torch

from .configuration import build_runtime, load_dataset
from .controller import ControllerConfig, ValueController, encode_state_actions
from .policies import DebugPolicy
from .specialist_data import write_records
from .train_specialist import train_from_file


def validate(config_path, output, limit=None, train=False, start_index=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    if config["runtime"]["experiment_mode"] != "real_debug":
        raise ValueError("Validation requires explicit uncalibrated debug mode")
    runtime = build_runtime(config)
    backend = runtime.backend.backend
    tasks = load_dataset("examples/debug_tasks.jsonl")[start_index:]
    if limit:
        tasks = tasks[:limit]
    identity = {"deployment": backend.deployment_identity,
                "models": {key: {"config": handle.model.config.to_dict(),
                    "hidden_dim": handle.hidden_dim, "chat_template": handle.codec.tokenizer.chat_template,
                    "thinking": handle.codec.chat_template_kwargs} for key, handle in backend.handles.items()},
                "torch": torch.__version__}
    (output/"model_manifest.json").write_text(json.dumps(identity, indent=2, default=str))
    dims = config["observation"]
    controller = ValueController(ControllerConfig(**dims)).eval()
    records, training_row = [], None
    modes = [("anchor_only", 1), ("self_repair", 1), ("help_submit", 1),
             ("help_expand_submit", 1), ("help_abort", 1), ("help_expand_submit", 2)]
    with (output/"traces.jsonl").open("w") as stream:
        for index, item in enumerate(tasks, start=start_index):
            mode, sessions = modes[index % len(modes)]
            started = time.perf_counter()
            state = runtime.initial(item.task, seed=7+index)
            legal = runtime.actions(state)
            forward = {"status": "no_legal_actions"}
            if legal:
                obs, actions = encode_state_actions(runtime, state, legal, **dims)
                with torch.no_grad():
                    values = controller(obs, actions)
                forward = {"status": "random_controller_forward_only", "legal": [a.key for a in legal],
                    "scores": {key: value.tolist() for key, value in values.items()},
                    "feature_dimensions": {key: len(value.values) for key, value in state.features.items()},
                    "feature_norms": {key: float(torch.tensor(value.values).norm()) for key, value in state.features.items()},
                    "help_request": asdict(state.help_request) if state.help_request else None}
            debug_policy = DebugPolicy(mode, sessions)
            phase_forwards = []

            def observed_policy(prefix, candidates):
                observation, encoded = encode_state_actions(runtime, prefix, candidates, **dims)
                with torch.no_grad():
                    scores = controller(observation, encoded)
                phase_forwards.append({"phase": "specialist" if prefix.help else "anchor",
                    "stage": prefix.help.stage if prefix.help else None,
                    "feature_dimensions": {key: len(feature.values) for key, feature in prefix.features.items()},
                    "legal": [action.key for action in candidates],
                    "scores": {key: value.tolist() for key, value in scores.items()}})
                return debug_policy(prefix, candidates)

            result = runtime.run_from(state, observed_policy)
            row = {"task": asdict(item.task), "mode": mode, "requested_sessions": sessions,
                "calls": result.calls, "steps": [asdict(step) for step in result.steps],
                "events": [asdict(event) for event in result.events], "decisions": result.decision_log,
                "assistance": result.assistance_log, "retained": result.artifacts,
                "candidate": result.candidate, "termination_reason": result.termination_reason,
                "tokens_by_model": result.ledger.tokens_by_model(), "cost_complete": False,
                "experiment_mode": "real_debug", "synthetic": False, "controller_forward": forward,
                "phase_forwards": phase_forwards,
                "e2e_ms": (time.perf_counter()-started)*1000,
                "peak_gpu_memory": {str(device): torch.cuda.max_memory_allocated(device) for device in (4, 7)}}
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
            print(json.dumps({"task": item.task.task_id, "mode": mode, "calls": result.calls,
                              "termination": result.termination_reason}), flush=True)
            if training_row is None:
                event = next((event for event in result.events if event.role == "specialist" and event.valid), None)
                if event:
                    training_row = {"input_id": "real-protocol-smoke", "prompt": event.prompt,
                                    "response": event.output, "synthetic": False}
    (output/"summary.json").write_text(json.dumps({"cases": len(records), "valid_calls": sum(event["valid"] for row in records for event in row["events"]),
        "invalid_calls": sum(not event["valid"] for row in records for event in row["events"]), "cost_complete": False}, indent=2))
    if train and training_row:
        del state, result, runtime, backend
        gc.collect()
        torch.cuda.empty_cache()
        write_records(output/"sft.jsonl", [training_row])
        spec = config["backend"]["models"]["A"]
        spec["chat_template_kwargs"] = {"enable_thinking": False}
        warm = train_from_file(output/"sft.jsonl", output/"sft_adapter", spec,
            steps=1, lora_rank=2, lora_alpha=4, max_length=4096, lr=2e-5)
        gc.collect()
        torch.cuda.empty_cache()
        chosen = json.loads(training_row["response"])
        rejected = json.loads(training_row["response"])
        rejected["draft"] = "An alternative unverified finding for loss-interface smoke testing."
        preference = {"input_id": training_row["input_id"], "prompt": training_row["prompt"],
            "chosen": json.dumps(chosen), "rejected": json.dumps(rejected), "synthetic": True,
            "label_source": "synthetic_pair_for_gradient_smoke_only_no_receiver_utility_claim"}
        write_records(output/"dpo_smoke.jsonl", [preference])
        trained = train_from_file(output/"dpo_smoke.jsonl", output/"dpo_adapter", spec, mode="dpo",
            init_adapter=str(output/"sft_adapter"), reference_adapter=str(output/"sft_adapter"),
            steps=1, max_length=4096, lr=2e-5)
        (output/"training_summary.json").write_text(json.dumps({"sft": warm, "dpo": trained}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_8b_debug.json")
    parser.add_argument("--output", default="outputs/qwen3/validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    validate(args.config, args.output, args.limit, args.train, args.start_index)


if __name__ == "__main__":
    main()
