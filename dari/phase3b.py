"""Paired collaboration interventions using frozen Phase 3A states."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random

import torch

from .configuration import build_runtime
from .phase2b_evaluation import EVALUATOR_VERSION, score
from .phase3a import ROOT as PREVIOUS, save_state, setup, write_json
from .phase3a_cost import MeasuredBackend
from .pilot import finish_or_continue, help_action, metrics, read_rows, state_id, trace
from .pilot_data import prepare
from .runtime import Runtime
from .schema import Action, Kind, Task

ROOT = PREVIOUS.parent / "phase3b_collaboration"
CONDITIONS = {
    "directed": ("directed", "guided"),
    "check": ("check", "guided"),
    "recompute": ("recompute", "guided"),
    "autonomous": ("autonomous", "guided"),
    "solver": ("solver", "independent"),
    "independent": ("directed", "independent"),
    "independent_compare": ("directed", "compare"),
    "autonomous_independent": ("autonomous", "independent"),
    "autonomous_independent_critical": ("autonomous", "independent"),
}


def run(limit, fresh=False, supplement=None):
    config = json.loads((PREVIOUS / "manifest.json").read_text())["config"]
    models = config["backend"]["models"]
    if supplement == "matched_sampling":
        for key in ("temperature", "top_p"):
            models["S"][key] = models["A"].get(key, 1.0)
    else:
        models["A"]["device_map"] = dict.fromkeys(models["A"]["device_map"], 0)
        models["S"]["device_map"] = {key: value - 1 for key, value in models["S"]["device_map"].items()}
        if supplement == "combined_validation":
            for layer in range(18, 25):
                models["S"]["device_map"][f"model.layers.{layer}"] = 1
    config["runtime"].update(experiment_mode="real_debug", max_specialist_stages=1)
    base = build_runtime(config)
    backend = MeasuredBackend(base.backend)
    rows = {row["task_id"]: row for split in setup(config).values() for row in split}
    paths = sorted((PREVIOUS / "collection/dev").rglob("prefix.pt"))
    paths += sorted((PREVIOUS / "collection/train").rglob("prefix.pt"))
    random.Random(20261128).shuffle(paths)
    states = []
    seen = set()
    for path in paths:
        prefix = torch.load(path, weights_only=False)
        if prefix.help or prefix.candidate or prefix.task.task_id in seen:
            continue
        if help_action(base, prefix, "narrow"):
            seen.add(prefix.task.task_id)
            states.append((path, prefix))
        if len(states) == limit:
            break
    experiment = supplement or ("fresh_validation" if fresh else "screen")
    conditions = [name for name in CONDITIONS if name != "autonomous_independent_critical"]
    if supplement == "matched_sampling":
        conditions = ["directed", "autonomous_independent"]
        paths = [Path(path) for path in
                 json.loads((ROOT / "manifest.json").read_text())["prefixes"]]
        states = [(path, torch.load(path, weights_only=False)) for path in paths]
    if fresh:
        from .phase3b_summary import choose
        conditions = (["directed", "autonomous_independent", "autonomous_independent_critical"]
                      if supplement else ["directed", *choose()])
        excluded = list(rows.values())
        for path in ("phase2a/samples.jsonl", "phase2a_refinement/r3/samples.jsonl", "phase2b/t1/samples.jsonl"):
            excluded += read_rows(ROOT.parent / path)
        if supplement:
            excluded += read_rows(ROOT / "fresh_validation/samples.jsonl")
        sample_seed = 20261130 if supplement else 20261129
        tasks = prepare(output=ROOT / experiment, per_dataset=8, seed=sample_seed, exclude=excluded)
        from .phase2b_data import normalized_hash
        excluded_hashes = {normalized_hash(row["question"]) for row in excluded}
        excluded_hashes.update(normalized_hash(row["question"]) for file in ("train", "validation")
                               for row in read_rows(ROOT.parent / "phase2b" / (file + ".jsonl")))
        states = []
        for index, row in enumerate(tasks):
            digest = normalized_hash(row["question"])
            if digest in excluded_hashes:
                raise ValueError(f"Duplicate fresh question: {row['task_id']}")
            excluded_hashes.add(digest)
            rows[row["task_id"]] = row
            path = ROOT / experiment / "prefixes" / (str(index) + ".pt")
            if path.exists():
                prefix = torch.load(path, weights_only=False)
            else:
                prefix = base.initial(Task(row["task_id"], row["question"], row["sources"]), sample_seed + index)
                save_state(path, prefix)
            if not prefix.candidate and help_action(base, prefix, "narrow"):
                states.append((path, prefix))
    manifest = {"config": config, "evaluator": EVALUATOR_VERSION, "conditions": conditions,
        "selection_seed": 20261128, "prefixes": [str(path) for path, _ in states],
        "backend": backend.deployment_identity, "cost_note": "Actual serial milliseconds. Phase 3A admission estimator used provisionally on new placement."}
    if supplement:
        manifest["cuda_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    manifest_path = ROOT / experiment / "manifest.json" if supplement else ROOT / ("fresh_manifest.json" if fresh else "manifest.json")
    if supplement and manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Supplement configuration differs from saved manifest")
    write_json(manifest_path, manifest)
    for _, prefix in states:
        key = state_id(prefix)
        row = rows[prefix.task.task_id]
        directory = ROOT / experiment / key

        def save(name, terminal, shared=prefix):
            write_json(directory / (name + ".json"), {
                "task_id": row["task_id"], "dataset": row["dataset"], "question": row["question"],
                "prefix_id": key, "seed": prefix.seed, "condition": name,
                "score": score(row["dataset"], terminal.candidate, row["gold"]),
                "answer": terminal.candidate, "future_ms": terminal.ledger.total - shared.ledger.total,
                "metrics": metrics(terminal, shared), "trace": trace(terminal, shared)})

        for name in ("continue", "repair", "extended", *conditions):
            path = directory / (name + ".json")
            if path.exists():
                continue
            print(key[:10], row["task_id"], name, flush=True)
            role, context = CONDITIONS.get(name, ("directed", "guided"))
            staged = context == "compare"
            runtime = Runtime(backend, base.profiles, replace(base.config,
                max_anchor_steps=len(prefix.steps)+2+4, specialist_role=role,
                specialist_context=context, max_specialist_stages=2 if staged else 1,
                integration_mode="critical" if name == "autonomous_independent_critical" else "direct",
                stage_output_budgets=(256, 256) if staged else (512,)))
            if name in CONDITIONS:
                state = runtime.step(prefix, help_action(runtime, prefix, "narrow"))
                if staged and Action(Kind.EXPAND) in runtime.actions(state):
                    state = runtime.step(state, Action(Kind.EXPAND))
                if not fresh and not supplement and name == "directed" and Action(Kind.SUBMIT) in runtime.actions(state):
                    for mode in ("abort", "direct", "critical", "comparison"):
                        if (directory / ("integration_" + mode + ".json")).exists():
                            continue
                        receiving = Runtime(backend, base.profiles, replace(runtime.config,
                            integration_mode="direct" if mode == "abort" else mode,
                            max_anchor_steps=len(state.steps)+5))
                        action = Action(Kind.ABORT if mode == "abort" else Kind.SUBMIT)
                        terminal = receiving.run_from(receiving.step(state, action), finish_or_continue)
                        save("integration_" + mode, terminal, state)
                decision = Action(Kind.SUBMIT) if Action(Kind.SUBMIT) in runtime.actions(state) else Action(Kind.ABORT)
                state = runtime.step(state, decision)
            else:
                state = runtime.step(prefix, Action(Kind.CONTINUE if name == "continue" else Kind.REPAIR))
                if name == "extended" and not state.terminal:
                    runtime._anchor(state, "Recheck the local result and advance the global solution.")
            continuation = Runtime(backend, base.profiles, replace(runtime.config, max_anchor_steps=len(state.steps)+4))
            save(name, continuation.run_from(state, finish_or_continue))
            assert state_id(prefix) == key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--supplement", choices=("matched_sampling", "combined_validation"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.limit, args.fresh or args.supplement == "combined_validation", args.supplement)
