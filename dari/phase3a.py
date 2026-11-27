"""Phase 3A selection, resumable paired rollouts, and closed-loop inference."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import torch

from .collection import collect_branches, deployment_id, write_targets
from .configuration import build_runtime
from .controller import ControllerConfig, NeuralPolicy, ValueController
from .identity import fingerprint
from .phase2b_data import normalized_hash
from .phase2b_evaluation import EVALUATOR_VERSION, score
from .phase3a_cost import MeasuredBackend, ROOT
from .pilot import metrics, read_rows, state_id, trace
from .pilot_data import prepare
from .runtime import Decision, Runtime
from .schema import Kind, Task

CONFIG = Path("configs/phase3a.json")
POLICY = "phase3a-v1/finish-continue/first-valid-submit-else-abort"
SPLITS = {"train": 20, "dev": 8, "eval": 8}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def setup(config):
    historical = [row for path in (
        "phase2a/samples.jsonl", "phase2a_refinement/r3/samples.jsonl", "phase2b/t1/samples.jsonl"
    ) for row in read_rows(ROOT.parent / path)]
    known = historical + read_rows(ROOT.parent / "phase2b/train.jsonl") + read_rows(ROOT.parent / "phase2b/validation.jsonl")
    hashes = {normalized_hash(row["question"]) for row in known}
    task_ids = {row["task_id"] for row in known}
    selection = {}
    for index, (split, per_dataset) in enumerate(SPLITS.items()):
        rows = prepare(output=ROOT / "splits" / split, seed=config["seed"] + index,
                       per_dataset=per_dataset, exclude=historical)
        for row in rows:
            digest = normalized_hash(row["question"])
            if digest in hashes or row["task_id"] in task_ids:
                raise ValueError(f"Overlapping task {row['task_id']}")
            hashes.add(digest)
            task_ids.add(row["task_id"])
        historical.extend(rows)
        selection[split] = [row["task_id"] for row in rows]
    write_json(ROOT / "splits/selection.json", selection)
    return {split: sorted(read_rows(ROOT / "splits" / split / "samples.jsonl"),
                          key=lambda row: (row["selection_rank"], row["dataset"])) for split in SPLITS}


def actions_for(state, legal):
    return [a for a in legal if a.kind != Kind.HELP or
            (a.width == "narrow" and a.quota == "short" and state.help_count == 0)]


def fixed(state, legal, mode="anchor", used_repair=False):
    if state.help:
        order = [Kind.SUBMIT, Kind.ABORT]
    elif mode == "bootstrap" and len(state.steps) >= 2 and not state.help_request.fallback:
        order = [Kind.FINISH, Kind.HELP, Kind.CONTINUE]
    elif mode == "help":
        order = [Kind.FINISH, Kind.HELP, Kind.CONTINUE]
    elif mode in {"repair", "extended"} and not used_repair:
        order = [Kind.FINISH, Kind.REPAIR, Kind.CONTINUE]
    else:
        order = [Kind.FINISH, Kind.CONTINUE]
    return next((a for kind in order for a in legal if a.kind == kind), legal[0])


def timed(policy, unit):
    def select(state, legal):
        started = time.perf_counter()
        action = policy(state, actions_for(state, legal))
        select.last_scores = getattr(policy, "last_scores", None)
        if isinstance(action, Decision):
            action = action.action
        return Decision(action, (time.perf_counter() - started) * 1000, unit)
    return select


class TaskEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def score(self, answer, gold):
        return score(self.dataset, answer, gold)["reward"]


class Experiment:
    def __init__(self, *, closed_only=False):
        config = json.loads(CONFIG.read_text())
        self.rows = setup(config)
        self.seed = config["seed"]
        calibration = json.loads((ROOT / "c0/calibration.json").read_text())
        self.complete = calibration["calibration_confirmed"]
        config.update(profiles=calibration["profiles"], calibration_confirmed=self.complete)
        config["runtime"].update(experiment_mode="formal" if self.complete else "real_debug",
                                 reference_cost=calibration["reference_cost"], budget=240000)
        raw = build_runtime(config)
        self.runtime = Runtime(MeasuredBackend(raw.backend), raw.profiles, raw.config)
        self.dims = tuple(config["observation"][key] for key in ("anchor_dim", "specialist_dim"))
        self.unit = calibration["unit"]
        self.continuation = timed(fixed, self.unit)
        manifest = {"config": config, "evaluator": EVALUATOR_VERSION, "policy": POLICY,
                    "deployment_id": deployment_id(self.runtime), "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "selection": json.loads((ROOT / "splits/selection.json").read_text()),
                    "collection": "First visited Anchor checkpoint and optional next selected state. Alternate bootstrap and Anchor trajectories. Finish only with candidate. C2 first valid narrow Help stage.",
                    "repeats": "Two rollouts for first three training tasks' Anchor alternatives. One otherwise.",
                    "cost_complete": self.complete}
        path = ROOT / "manifest.json"
        if closed_only:
            collection_manifest = json.loads(path.read_text())
            comparable = {**manifest, "cuda_devices": collection_manifest["cuda_devices"]}
            if comparable != collection_manifest:
                raise ValueError("Closed-loop configuration differs from collection")
        else:
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError("Existing experiment manifest mismatch")
            write_json(path, manifest)

    def collect_group(self, row, split, prefix, parent, repeats):
        runtime = self.runtime
        prefix_id = state_id(prefix)
        phase = "specialist" if prefix.help else "anchor"
        directory = ROOT / "collection" / split / row["task_id"].replace(":", "_") / prefix_id
        group = directory / "targets.jsonl"
        if group.exists():
            return
        path = directory / "prefix.pt"
        if not path.exists():
            save_state(path, prefix)
        legal = runtime.actions(prefix)
        actions = actions_for(prefix, legal)

        def rollout(branch, action, repeat):
            key = fingerprint({"action": asdict(action), "repeat": repeat})[:20]
            saved = directory / f"{key}.pt"
            if saved.exists():
                first, terminal = torch.load(saved, weights_only=False)
            else:
                first = runtime.step(branch, action)
                terminal = runtime.run_from(first, self.continuation)
                save_state(saved, (first, terminal))
            if (phase == "anchor" and action.kind == Kind.HELP and repeat == 0
                    and any(a.kind == Kind.SUBMIT for a in runtime.actions(first))):
                self.collect_group(row, split, first, prefix_id, 1)
            if terminal.ledger.entries[:len(prefix.ledger.entries)] != prefix.ledger.entries:
                raise RuntimeError("Shared sunk cost changed")
            if phase == "specialist" and action.kind == Kind.ABORT:
                from .context import anchor_input
                from .runtime import CONTINUE_INSTRUCTION
                expected = anchor_input(prefix, CONTINUE_INSTRUCTION).text
                events = terminal.events[len(prefix.events):]
                if events and events[0].prompt != expected:
                    raise RuntimeError("Abort prompt differs from plain Anchor continuation")
            write_json(directory / f"{key}.json", {
                "action": asdict(action), "seed": branch.seed, "repeat": repeat,
                "reward": score(row["dataset"], terminal.candidate, row["gold"]),
                "answer": terminal.candidate, "future_ms": terminal.ledger.total - prefix.ledger.total,
                "shared_ms": prefix.ledger.total,
                "metrics": {**metrics(terminal, prefix), "cost_complete": self.complete},
                "trace": trace(terminal, prefix), "termination_reason": terminal.termination_reason,
            })
            return terminal

        print("COLLECT", split, row["task_id"], phase, [a.kind.value for a in actions], flush=True)
        targets = collect_branches(runtime, prefix, actions, self.continuation, TaskEvaluator(row["dataset"]),
                                   row["gold"], repeats, POLICY, *self.dims, rollout=rollout,
                                   reward_only=not self.complete)
        if state_id(prefix) != prefix_id:
            raise RuntimeError("Shared state mutated")
        targets = [replace(target, prefix_id=prefix_id, provenance={
            **target.provenance, "task_id": row["task_id"], "dataset": row["dataset"], "split": split,
            "parent_state_id": parent, "phase": phase, "legal_actions": [a.key for a in legal],
            "sampled_actions": [a.key for a in actions], "candidate_present": prefix.candidate is not None,
            "help_stage": prefix.help.stage if prefix.help else 0,
            "has_prior_assistance": bool(prefix.artifacts or prefix.assistance_log),
            "prefix_path": str(path), "reward_variance": target.reward_std ** 2 if repeats > 1 else None,
            "cost_variance": target.cost_std ** 2 if repeats > 1 else None,
        }) for target in targets]
        tmp = group.with_suffix(".tmp")
        write_targets(tmp, targets)
        tmp.replace(group)

    def collect(self, split, start, stop):
        for index, row in enumerate(self.rows[split][start:stop], start):
            runtime = self.runtime
            directory = ROOT / "visits" / split / row["task_id"].replace(":", "_")
            path = directory / "prefixes.pt"
            if path.exists():
                prefixes = torch.load(path, weights_only=False)
            else:
                state = runtime.initial(Task(row["task_id"], row["question"], row["sources"]), self.seed + index)
                prefixes = []
                if runtime.actions(state):
                    prefixes.append(state)
                    if state.candidate is None:
                        follow = timed(lambda s, a: fixed(s, a, "help" if index % 2 else "anchor"), self.unit)
                        later = runtime.decide_step(state, follow)
                        if later.help and runtime.actions(later):
                            later = runtime.decide_step(later, self.continuation)
                        if runtime.actions(later):
                            prefixes.append(later)
                save_state(path, prefixes)
                write_json(directory / "visit.json", {"task_id": row["task_id"], "prefixes": [state_id(p) for p in prefixes],
                                                      "initial_trace": trace(state)})
            for index_in_task, prefix in enumerate(prefixes):
                self.collect_group(row, split, prefix, state_id(prefixes[0]) if index_in_task else None,
                                   2 if split == "train" and index < 3 else 1)

    def closed(self, limit):
        checkpoint = ROOT / "training/full.pt"
        saved = torch.load(checkpoint, weights_only=True)
        if saved["data_metadata"]["deployment_id"] != deployment_id(self.runtime):
            raise ValueError("Controller checkpoint deployment differs from inference")
        model = ValueController(ControllerConfig(**saved["config"]))
        model.load_state_dict(saved["model_state"])
        learned = NeuralPolicy(self.runtime, model)
        for index, row in enumerate(self.rows["eval"][:limit]):
            for mode in ("anchor", "repair", "help", "bootstrap", "learned", "extended"):
                path = ROOT / "closed" / row["task_id"].replace(":", "_") / f"{mode}.json"
                if path.exists():
                    continue
                print("CLOSED", row["task_id"], mode, flush=True)
                used_repair = False

                def policy(state, legal):
                    nonlocal used_repair
                    action = fixed(state, legal, mode, used_repair and (mode != "extended" or
                                   sum(d["selected_action"].split("|")[0] == "self_repair" for d in state.decision_log) >= 2))
                    used_repair = used_repair or action.kind == Kind.REPAIR
                    return action

                state = self.runtime.run(Task(row["task_id"], row["question"], row["sources"]),
                    timed(learned if mode == "learned" else policy, self.unit), self.seed + 1000 + index)
                write_json(path, {"task_id": row["task_id"], "mode": mode,
                    "seed": state.seed, "answer": state.candidate,
                    "score": score(row["dataset"], state.candidate, row["gold"]),
                    "ms": state.ledger.total,
                    "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "metrics": {**metrics(state), "cost_complete": self.complete},
                    "trace": trace(state)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["setup", "collect", "closed"])
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "setup":
        setup(json.loads(CONFIG.read_text()))
    elif args.stage == "collect":
        Experiment().collect(args.split, args.start, args.stop)
    else:
        Experiment(closed_only=True).closed(args.stop)


if __name__ == "__main__":
    main()
