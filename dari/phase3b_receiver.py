"""Run receiver interventions on saved Specialist states using Anchor only."""
from dataclasses import replace
import json
import os

import torch

from .configuration import RecordingBackend
from .costs import ModelProfile
from .hf_backend import HuggingFaceBackend
from .phase2b_evaluation import EVALUATOR_VERSION, score
from .phase3a import write_json
from .phase3a_cost import MeasuredBackend
from .phase3b import PREVIOUS, ROOT
from .pilot import finish_or_continue, metrics, read_rows, state_id, trace
from .runtime import Runtime, RuntimeConfig
from .schema import Action, Kind


class ReceiverBackend(MeasuredBackend):
    def count_tokens(self, model_id, text):
        return super().count_tokens("A" if model_id == "S" else model_id, text)

    def generate(self, request):
        if request.model_id != "A":
            raise ValueError("Receiver replay only permits Anchor generation")
        return super().generate(request)


def main():
    torch.set_num_threads(2)
    output = ROOT / "integration_history"
    config = json.loads((PREVIOUS / "manifest.json").read_text())["config"]
    config["backend"]["models"].pop("S")
    config["runtime"].update(experiment_mode="real_debug", max_specialist_stages=1)
    tasks = {row["task_id"]: row for row in read_rows(PREVIOUS / "splits/dev/samples.jsonl")}
    prefixes = []
    seen = set()
    for path in sorted((PREVIOUS / "collection/dev").rglob("prefix.pt")):
        prefix = torch.load(path, weights_only=False, map_location="cpu")
        if prefix.help and prefix.help.draft and prefix.task.task_id not in seen:
            prefixes.append((path, prefix))
            seen.add(prefix.task.task_id)
    manifest = {"run_id": "integration_history", "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "config": config, "evaluator": EVALUATOR_VERSION,
                "selection": "First saved valid Specialist state per development task, sorted by path",
                "prefixes": [str(path) for path, _ in prefixes]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Receiver run identity changed")
    write_json(manifest_path, manifest)
    backend = ReceiverBackend(RecordingBackend(HuggingFaceBackend.from_specs(config["backend"]["models"])))
    profiles = {profile["model_id"]: ModelProfile(**profile) for profile in config["profiles"]}
    runtime_config = RuntimeConfig(**config["runtime"])
    for _, prefix in prefixes:
        key = state_id(prefix)
        task = tasks[prefix.task.task_id]
        for mode in ("abort", "direct", "critical", "comparison"):
            path = output / key / ("integration_" + mode + ".json")
            if path.exists():
                continue
            print(key, task["task_id"], mode, flush=True)
            runtime = Runtime(backend, profiles, replace(runtime_config,
                integration_mode="direct" if mode == "abort" else mode,
                max_anchor_steps=len(prefix.steps)+5))
            action = Action(Kind.ABORT if mode == "abort" else Kind.SUBMIT)
            terminal = runtime.run_from(runtime.step(prefix, action), finish_or_continue)
            write_json(path, {"condition": "integration_" + mode, "prefix_id": key,
                "task_id": task["task_id"], "question": task["question"], "seed": prefix.seed,
                "draft": prefix.help.draft.text, "answer": terminal.candidate,
                "score": score(task["dataset"], terminal.candidate, task["gold"]),
                "future_ms": terminal.ledger.total-prefix.ledger.total,
                "metrics": metrics(terminal, prefix), "trace": trace(terminal, prefix)})


if __name__ == "__main__":
    main()
