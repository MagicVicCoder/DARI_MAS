"""Offline branch targets and receiver-utility preference records."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Callable, Sequence

from .controller import encode_state_actions
from .evaluation import Evaluator
from .runtime import EpisodeState, Runtime
from .schema import Action, Kind


@dataclass(frozen=True)
class ControllerTarget:
    prefix_id: str
    action: Action
    reward: float
    cost: float
    reward_std: float
    cost_std: float
    observation: dict
    action_features: dict
    snapshot_id: str
    cost_unit: str
    synthetic: bool
    repeats: int
    reference_cost: float
    deployment_id: str
    cost_complete: bool
    base_deployment_id: str = ""


def deployment_id(runtime: Runtime, include_specialist_adapter: bool = True) -> str:
    identity = deepcopy(runtime.backend.deployment_identity)
    if not include_specialist_adapter:
        identity.get("adapters", {}).pop(runtime.config.specialist_model, None)
    encoded = json.dumps({"config": asdict(runtime.config),
        "profiles": {k: asdict(v) for k, v in runtime.profiles.items()}, "backend": identity}, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collect_branches(runtime: Runtime, prefix: EpisodeState, actions: Sequence[Action],
                     policy: Callable, evaluator: Evaluator, gold: str, repeats: int,
                     snapshot_id: str, anchor_dim: int, specialist_dim: int,
                     allow_unmeasured_overhead: bool = False) -> list[ControllerTarget]:
    if repeats <= 0 or not snapshot_id:
        raise ValueError("Positive repeats and explicit policy/S snapshot ID required")
    if any(a not in runtime.actions(prefix) for a in actions):
        raise ValueError("Cannot collect an illegal alternative")
    # Gold/evaluator live outside this digest and outside model observations.
    encoded_prefix = json.dumps(asdict(prefix), sort_keys=True, ensure_ascii=False)
    prefix_id = hashlib.sha256(encoded_prefix.encode("utf-8")).hexdigest()
    deployment = deployment_id(runtime)
    rows = []
    for action in actions:
        returns, costs = [], []
        complete = True
        for repeat in range(repeats):
            branch = deepcopy(prefix)
            branch.seed = prefix.seed + repeat
            branch = runtime.step(branch, action)
            terminal = runtime.run_from(branch, policy)
            complete = complete and len(terminal.ledger.unmeasured_overheads) == len(prefix.ledger.unmeasured_overheads)
            if (not allow_unmeasured_overhead and
                    len(terminal.ledger.unmeasured_overheads) > len(prefix.ledger.unmeasured_overheads)):
                raise ValueError("Unmeasured controller overhead; provide calibrated Decision cost")
            reward = evaluator.score(terminal.candidate, gold)
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError("Evaluator must return a normalized score")
            returns.append(reward)
            costs.append((terminal.ledger.total-prefix.ledger.total)/runtime.config.reference_cost)
        obs, encoded_action = encode_state_actions(runtime, prefix, [action], anchor_dim, specialist_dim)
        rows.append(ControllerTarget(prefix_id, action, statistics.mean(returns), statistics.mean(costs),
            statistics.stdev(returns) if repeats > 1 else 0., statistics.stdev(costs) if repeats > 1 else 0.,
            {k: v.tolist() for k, v in obs.items()}, {k: v.tolist() for k, v in encoded_action.items()},
            snapshot_id, prefix.ledger.unit, runtime.backend.synthetic, repeats,
            runtime.config.reference_cost, deployment, complete, deployment_id(runtime, False)))
    return rows


def write_targets(path: str | Path, rows: Sequence[ControllerTarget]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            body = asdict(row)
            body["schema_version"] = 1
            handle.write(json.dumps(body, ensure_ascii=False) + "\n")


def read_targets(path: str | Path) -> list[ControllerTarget]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            body = json.loads(line)
            if body.pop("schema_version", None) != 1:
                raise ValueError("Unsupported target schema")
            body["action"]["kind"] = Kind(body["action"]["kind"])
            body["action"] = Action(**body["action"])
            rows.append(ControllerTarget(**body))
    return rows


@dataclass(frozen=True)
class StageOutcome:
    """input_id must hash exact S prompt, quota, history and policy snapshots."""
    input_id: str
    output: str
    utility: float
    standard_error: float


@dataclass(frozen=True)
class PreferencePair:
    input_id: str
    chosen: str
    rejected: str
    gap: float


def preference_pairs(outcomes: Sequence[StageOutcome], min_gap: float = .01,
                     uncertainty_multiplier: float = 2.) -> list[PreferencePair]:
    groups: dict[str, list[StageOutcome]] = {}
    for item in outcomes:
        if not math.isfinite(item.utility) or not math.isfinite(item.standard_error) or item.standard_error < 0:
            raise ValueError("Invalid utility estimate")
        groups.setdefault(item.input_id, []).append(item)
    pairs = []
    for input_id, items in groups.items():
        for first, second in itertools.combinations(items, 2):
            chosen, rejected = sorted((first, second), key=lambda x: x.utility, reverse=True)
            threshold = max(min_gap, uncertainty_multiplier*math.hypot(chosen.standard_error, rejected.standard_error))
            if chosen.output != rejected.output and chosen.utility-rejected.utility > threshold:
                pairs.append(PreferencePair(input_id, chosen.output, rejected.output, chosen.utility-rejected.utility))
    return pairs
