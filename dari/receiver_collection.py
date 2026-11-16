"""Phi labels from executed receiver continuations, with exact one-call replay."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import statistics

from .backends import Generation, GenerationRequest
from .collection import StageOutcome, deployment_id, preference_pairs
from .runtime import Runtime
from .schema import Action, Kind, SpecialistRecord


class ReplayOneBackend:
    """Substitute an already sampled S output once; all later calls really run.

    Sampling/training accounting belongs to the underlying recording backend.
    The replayed generation still appears in the branch's hypothetical online
    cost. A request mismatch is an error, never an accidental replay elsewhere.
    """
    def __init__(self, backend, expected: GenerationRequest, generation: Generation):
        self.backend, self.expected, self.generation = backend, expected, generation
        self.synthetic, self.used = backend.synthetic, False

    @property
    def deployment_identity(self):
        return self.backend.deployment_identity

    def count_tokens(self, model_id, text):
        return self.backend.count_tokens(model_id, text)

    def count_text_tokens(self, model_id, text):
        return self.backend.count_text_tokens(model_id, text)

    def generate(self, request):
        if not self.used:
            if replace(request, seed=0) != replace(self.expected, seed=0):
                raise ValueError("Sampled output replay received a different model input")
            self.used = True
            return self.generation
        return self.backend.generate(request)


@dataclass
class SpecialistCollection:
    preferences: list[dict]
    auxiliary: list[dict]
    outcomes: list[dict]


def _score(terminal, evaluator, gold):
    reward = float(evaluator.score(terminal.candidate, gold))
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Evaluator must return a finite normalized score")
    return reward


def _check_overhead(terminal, prefix):
    if len(terminal.ledger.unmeasured_overheads) > len(prefix.ledger.unmeasured_overheads):
        raise ValueError("Receiver utility requires complete, compatible controller cost labels")


def collect_specialist(runtime: Runtime, prefix, action: Action, policy, evaluator, gold: str,
                       candidates: int = 4, repeats: int = 2, snapshot_id: str = "",
                       lambda_cost: float = .1, min_gap: float = .01,
                       early_probe: bool = False) -> SpecialistCollection:
    if candidates < 2 or repeats <= 0 or not snapshot_id:
        raise ValueError("At least two candidates, positive repeats and a frozen snapshot ID required")
    if not math.isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("Nonnegative finite cost weight required")
    if action.kind not in (Kind.HELP, Kind.EXPAND) or action not in runtime.actions(prefix):
        raise ValueError("Phi collection needs one legal help/expand stage")
    request = runtime.request_for_action(prefix, action)
    assert request is not None
    signature = {"request": asdict(replace(request, seed=0)), "action": action.key,
                 "snapshot": snapshot_id, "deployment": deployment_id(runtime), "lambda_cost": lambda_cost}
    input_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    metadata = {"input_id": input_id, "prompt": request.prompt, "snapshot_id": snapshot_id,
                "deployment_id": deployment_id(runtime), "cost_unit": prefix.ledger.unit,
                "reference_cost": runtime.config.reference_cost, "synthetic": runtime.backend.synthetic,
                "max_output_tokens": request.max_new_tokens, "draft_limit": runtime.config.draft_limit,
                "target": action.target if action.kind == Kind.HELP else prefix.help.action.target}
    outcomes, auxiliary, valid_outputs = [], [], set()
    for sample in range(candidates):
        sampled = runtime.backend.generate(replace(request, seed=prefix.seed+sample))
        if sampled.usage.model_id != request.model_id:
            raise ValueError("Sample backend returned the wrong model usage")
        try:
            record = SpecialistRecord.parse(sampled.text)
            valid = (sampled.usage.output_tokens <= request.max_new_tokens and
                     record.draft.target == metadata["target"] and
                     runtime.backend.count_text_tokens(request.model_id, record.draft.to_json()) <= runtime.config.draft_limit)
        except ValueError:
            valid = False
        if valid:
            valid_outputs.add(sampled.text)
        utilities, early_gains = [], []
        for repeat in range(repeats):
            branch = deepcopy(prefix)
            branch.seed = prefix.seed+100003+repeat
            replay = ReplayOneBackend(runtime.backend, request, sampled)
            branch_runtime = Runtime(replay, runtime.profiles, runtime.config)
            helped = branch_runtime.step(branch, action)
            terminal = branch_runtime.run_from(helped, policy)
            _check_overhead(terminal, prefix)
            future_cost = (terminal.ledger.total-prefix.ledger.total)/runtime.config.reference_cost
            utilities.append(_score(terminal, evaluator, gold)-lambda_cost*future_cost)
            if early_probe and valid and Action(Kind.SUBMIT) in branch_runtime.actions(helped):
                submitted = branch_runtime.run_from(branch_runtime.step(helped, Action(Kind.SUBMIT)), policy)
                aborted = branch_runtime.run_from(branch_runtime.step(helped, Action(Kind.ABORT)), policy)
                _check_overhead(submitted, helped)
                _check_overhead(aborted, helped)
                difference = _score(submitted, evaluator, gold)-_score(aborted, evaluator, gold)
                extra_cost = (submitted.ledger.total-aborted.ledger.total)/runtime.config.reference_cost
                early_gains.append(difference-lambda_cost*extra_cost)
        standard_error = statistics.stdev(utilities)/math.sqrt(repeats) if repeats > 1 else 0.
        outcome = StageOutcome(input_id, sampled.text, statistics.mean(utilities), standard_error)
        outcomes.append(outcome)
        # Forced Submit is a separate local usefulness test. It never overwrites
        # the main policy-return target or asserts monotonically improving drafts.
        if early_gains and statistics.mean(early_gains) > min_gap:
            auxiliary.append({**metadata, "response": sampled.text, "source": "early_submit_gain",
                              "gain": statistics.mean(early_gains)})
    preferences = [{**metadata, "chosen": p.chosen, "rejected": p.rejected, "gap": p.gap}
                   for p in preference_pairs(outcomes, min_gap=min_gap)
                   if p.chosen in valid_outputs and p.rejected.strip()]
    return SpecialistCollection(preferences, auxiliary, [asdict(o) for o in outcomes])
