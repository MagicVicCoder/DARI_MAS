"""CLI adapter for the native Runtime."""
from dataclasses import asdict
import json
from pathlib import Path

from .configuration import evaluator_for, load_dataset
from .hf_backend import HuggingFaceBackend
from .native import NativeAction, NativeConfig, NativeRuntime


def run_native_dataset(config, tasks, output, evaluate, trace):
    records = load_dataset(tasks, require_gold=evaluate)
    backend_config = config["backend"]
    if backend_config["type"] != "hf":
        raise ValueError("Native CLI requires a Hugging Face backend")
    if config.get("controller_checkpoint"):
        raise ValueError("Legacy Controller checkpoints are incompatible with native observations")
    models = {name: dict(spec) for name, spec in backend_config["models"].items()}
    for name, adapter in config.get("model_adapters", {}).items():
        models[name]["adapter"] = adapter
    runtime = NativeRuntime(HuggingFaceBackend.from_specs(models), NativeConfig(**config.get("native", {})))
    policy_name = config.get("native_policy", "anchor_only")
    if policy_name not in {"standard", "anchor_only", "help_once", "forced"}:
        raise ValueError("Unknown native policy")
    forced = NativeAction(**config["forced_action"]) if policy_name == "forced" else None

    def policy(state, legal):
        if state.session:
            return NativeAction("submit") if NativeAction("submit") in legal else NativeAction("abort")
        if not any(event["kind"] == "specialist" or event.get("action", {}).get("kind") == "self_repair"
                   for event in state.events):
            if forced in legal:
                return forced
            if policy_name == "help_once":
                return next((action for action in legal if action.kind == "help"), NativeAction("continue"))
        return NativeAction("continue")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(records):
            seed = config.get("seed", 7) + index
            state = (runtime.standard(record.task, seed) if policy_name == "standard"
                     else runtime.run(record.task, policy, seed))
            row = {"task_id": record.task.task_id, "answer": state.candidate,
                   "reasoning_mode": "native", "termination_reason": state.termination,
                   "anchor_tokens": len(state.generated_ids), "specialist_tokens": state.specialist_tokens,
                   "injection_tokens": sum(len(item["token_ids"]) for item in state.injections),
                   "cost_complete": False, "synthetic": False}
            if evaluate:
                row["score"] = evaluator_for(record.metric).score(state.candidate, record.gold)
            if trace:
                row.update(events=state.events, context_ids=state.context_ids, generated_ids=state.generated_ids,
                           prompt_ids=state.prompt_ids, observations=state.observations,
                           checkpoints=[asdict(item) for item in state.checkpoints], injections=state.injections)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows
