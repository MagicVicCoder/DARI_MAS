"""Bounded offline collection phases used by the CLI and alternating trainer."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

from .collection import collect_branches, write_targets
from .configuration import build_runtime, evaluator_for, fixed_overhead, make_policy
from .identity import artifact_digest
from .receiver_collection import collect_specialist
from .runtime import Decision
from .schema import Kind
from .specialist_data import write_records


def snapshot_id(config, controller, adapter):
    encoded = json.dumps({"config": config, "theta": artifact_digest(controller),
                          "phi": artifact_digest(adapter)}, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _subset(legal, limit, rng):
    if limit <= 0:
        raise ValueError("actions_per_prefix must be positive")
    if len(legal) <= limit:
        return legal
    # Preserve action-type/quota coverage before adding random alternative packs.
    result, groups = [], set()
    for action in legal:
        group = (action.kind, action.quota)
        if group not in groups:
            result.append(action)
            groups.add(group)
    if len(result) > limit:
        return rng.sample(result, limit)
    rest = [a for a in legal if a not in result]
    return result+rng.sample(rest, min(limit-len(result), len(rest)))


def _advance(runtime, state, policy, rng, epsilon, overhead):
    legal = runtime.actions(state)
    if not legal:
        return None
    if rng.random() < epsilon:
        chosen = rng.choice(legal)
        choice = Decision(chosen, *overhead()) if overhead else Decision(chosen)
        return runtime.decide_step(state, lambda *_: choice)
    return runtime.decide_step(state, policy)


def collect_phase(config, tasks, output: str | Path, mode: str,
                  controller: str | Path | None = None, adapter: str | Path | None = None,
                  allow_specialist_update: bool = False) -> dict:
    if mode not in {"controller", "specialist"}:
        raise ValueError("Expected controller or specialist collection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    active = deepcopy(config)
    if adapter:
        active.setdefault("model_adapters", {})[config.get("runtime", {}).get("specialist_model", "S")] = str(adapter)
    runtime = build_runtime(active)
    controller = controller or config.get("controller_checkpoint")
    policy = make_policy(runtime, active, controller, allow_specialist_update=allow_specialist_update)
    overhead = fixed_overhead(runtime, active)
    settings = config.get("training", {}).get("collection", {})
    repeats = settings.get("repeats", 2)
    prefix_limit = settings.get("prefixes_per_task", 4)
    epsilon = settings.get("epsilon", .2)
    if prefix_limit <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("Positive prefix limit and epsilon in [0,1] required")
    rng = random.Random(config.get("seed", 7))
    snapshot = snapshot_id(active, controller, adapter)
    rows, preferences, auxiliary, outcomes = [], [], [], []
    started = time.perf_counter()
    for number, item in enumerate(tasks):
        if item.gold is None:
            raise ValueError("Collection requires offline gold; it is never added to Task")
        state = runtime.initial(item.task, config.get("seed", 7)+number)
        evaluator = evaluator_for(item.metric)
        for _ in range(prefix_limit):
            legal = runtime.actions(state)
            if not legal:
                break
            if mode == "controller":
                selected = _subset(legal, settings.get("actions_per_prefix", 6), rng)
                dims = config["observation"]
                rows.extend(collect_branches(runtime, state, selected, policy, evaluator, item.gold,
                    repeats, snapshot, dims["anchor_dim"], dims["specialist_dim"]))
            else:
                # One exact input/target/quota per prefix; alternatives are output
                # samples of this input, never a cross-prompt DPO comparison.
                stage = next((a for a in legal if a.kind == Kind.EXPAND), None)
                help_actions = [action for action in legal if action.kind == Kind.HELP]
                if stage is None and help_actions:
                    stage = rng.choice(help_actions)
                if stage:
                    result = collect_specialist(runtime, state, stage, policy, evaluator, item.gold,
                        candidates=settings.get("candidates", 4), repeats=repeats, snapshot_id=snapshot,
                        lambda_cost=config.get("training", {}).get("controller", {}).get("lambda_cost", .1),
                        min_gap=settings.get("min_gap", .01), early_probe=settings.get("early_probe", False))
                    preferences.extend(result.preferences)
                    auxiliary.extend(result.auxiliary)
                    outcomes.extend(result.outcomes)
            state = _advance(runtime, state, policy, rng, epsilon, overhead)
            if state is None or state.terminal:
                break
    if mode == "controller":
        write_targets(output/"targets.jsonl", rows)
    else:
        write_records(output/"preferences.jsonl", preferences)
        write_records(output/"auxiliary.jsonl", auxiliary)
        write_records(output/"outcomes.jsonl", outcomes)
    report = {"mode": mode, "snapshot_id": snapshot, "controller_examples": len(rows),
              "preference_pairs": len(preferences), "auxiliary_examples": len(auxiliary),
              "collection_seconds": time.perf_counter()-started, **runtime.backend.report(runtime.profiles)}
    (output/"collection_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
