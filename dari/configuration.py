"""Model factories, offline dataset separation and deployment-checked policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path

import torch

from .backends import ScriptedBackend
from .collection import deployment_id
from .controller import ControllerConfig, NeuralPolicy, ValueController
from .costs import CostLedger, ModelProfile
from .evaluation import AnswerF1, ChoiceAccuracy, ExactMatch
from .runtime import Decision, Runtime, RuntimeConfig
from .schema import Kind, Task


def import_factory(name: str):
    module, separator, member = name.partition(":")
    if not separator or not module or not member:
        raise ValueError("Factory must use importable_module:function notation")
    factory = getattr(importlib.import_module(module), member)
    if not callable(factory):
        raise ValueError("Factory is not callable")
    return factory


@dataclass(frozen=True)
class LabeledTask:
    task: Task
    gold: str | None
    metric: str = "exact"


def load_dataset(path: str | Path, require_gold: bool = False) -> list[LabeledTask]:
    result, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            body = json.loads(line)
            task = Task(body["task_id"], body["question"], body.get("sources", {}))
            if (not isinstance(task.task_id, str) or not task.task_id or task.task_id in seen or
                    not isinstance(task.question, str) or not task.question.strip()):
                raise ValueError("Dataset requires unique IDs and nonempty questions")
            if not isinstance(task.sources, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                       for k, v in task.sources.items()):
                raise ValueError("sources must map string IDs to original text")
            gold = body.get("gold")
            if (require_gold and gold is None) or (gold is not None and not isinstance(gold, str)):
                raise ValueError("Offline labels must be strings; required gold is missing")
            seen.add(task.task_id)
            result.append(LabeledTask(task, gold, body.get("metric", "exact")))
    if not result:
        raise ValueError("Dataset contains no tasks")
    return result


def evaluator_for(metric: str):
    builtins = {"exact": ExactMatch, "choice": ChoiceAccuracy, "f1": AnswerF1}
    if metric in builtins:
        return builtins[metric]()
    # A factory may return an adapter around the official benchmark scorer.
    scorer = import_factory(metric)()
    if not callable(getattr(scorer, "score", None)):
        raise ValueError("Evaluator factory must return an object with score(answer, gold)")
    return scorer


class RecordingBackend:
    """Records actual calls for offline cost reporting, including discarded samples."""
    def __init__(self, backend):
        self.backend, self.synthetic = backend, backend.synthetic
        self.usage, self.policy_overheads = [], []

    @property
    def deployment_identity(self):
        return self.backend.deployment_identity

    def count_tokens(self, model_id, text):
        return self.backend.count_tokens(model_id, text)

    def count_text_tokens(self, model_id, text):
        return self.backend.count_text_tokens(model_id, text)

    def generate(self, request):
        generation = self.backend.generate(request)
        self.usage.append(generation.usage)
        return generation

    def report(self, profiles):
        ledger = CostLedger(profiles)
        for usage in self.usage:
            ledger.charge(usage)
        for cost, unit in self.policy_overheads:
            ledger.charge_overhead("policy", cost, unit)
        return {"inference_cost": ledger.total, "cost_unit": ledger.unit,
                "tokens_by_model": ledger.tokens_by_model(), "synthetic": self.synthetic,
                "scope": "Actual rollout model calls and declared policy overhead; optimization/reference forward reported separately"}


def build_runtime(config: dict) -> Runtime:
    spec = deepcopy(config["backend"])
    profiles = {p["model_id"]: ModelProfile(**p) for p in config["profiles"]}
    CostLedger(profiles)
    if any(not p.synthetic for p in profiles.values()) and not config.get("calibration_confirmed", False) and config.get("runtime", {}).get("experiment_mode") != "real_debug":
        raise ValueError("Real inference requires calibrated profiles and calibration_confirmed=true")
    kind = spec.pop("type")
    if kind == "hf":
        from .hf_backend import HuggingFaceBackend
        models = spec.pop("models")
        for model_id, adapter in config.get("model_adapters", {}).items():
            models[model_id]["adapter"] = adapter
        backend = HuggingFaceBackend.from_specs(models)
    elif kind == "factory":
        backend = import_factory(spec.pop("entrypoint"))(deepcopy(config))
    elif kind == "scripted":
        backend = ScriptedBackend(spec.pop("outputs"), spec.pop("dimensions", None))
    else:
        raise ValueError(f"Unknown backend type: {kind}")
    if spec:
        raise ValueError(f"Unknown backend options: {sorted(spec)}")
    for method in ("count_tokens", "count_text_tokens", "generate"):
        if not callable(getattr(backend, method, None)):
            raise ValueError(f"Backend must implement {method}")
    if hasattr(backend, "handles"):
        runtime_config = config.get("runtime", {})
        config["observation"] = {
            "anchor_dim": backend.handles[runtime_config.get("anchor_model", "A")].hidden_dim,
            "specialist_dim": backend.handles[runtime_config.get("specialist_model", "S")].hidden_dim}
    return Runtime(RecordingBackend(backend), profiles, RuntimeConfig(**config.get("runtime", {})))


def fixed_overhead(runtime: Runtime, config: dict):
    spec = config.get("policy_overhead")
    if spec is None:
        return None
    if not spec.get("provenance"):
        raise ValueError("Policy overhead requires a calibration provenance")
    # Validate units and values before any policy is executed.
    ledger = CostLedger(runtime.profiles)
    ledger.charge_overhead("policy", spec["cost"], spec["unit"])

    def overhead():
        if isinstance(runtime.backend, RecordingBackend):
            runtime.backend.policy_overheads.append((spec["cost"], spec["unit"]))
        return spec["cost"], spec["unit"]
    return overhead


def make_policy(runtime: Runtime, config: dict, checkpoint: str | Path | None = None,
                allow_specialist_update: bool = False):
    overhead = fixed_overhead(runtime, config)
    if config.get("debug_policy"):
        from .policies import DebugPolicy
        return DebugPolicy(config["debug_policy"], config.get("debug_help_sessions", 1))
    checkpoint = checkpoint or config.get("controller_checkpoint")
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("config", {}).get("schema_version") != 2:
            raise ValueError("Legacy Controller checkpoint is incompatible with dynamic assistance v2")
        metadata = saved["data_metadata"]
        same_deployment = metadata["deployment_id"] == deployment_id(runtime)
        permitted_phi_update = (allow_specialist_update and metadata.get("base_deployment_id") == deployment_id(runtime, False))
        if (not (same_deployment or permitted_phi_update) or metadata["synthetic"] != runtime.backend.synthetic):
            raise ValueError("Controller deployment/cost profile differs; recollect or explicitly retrain")
        model = ValueController(ControllerConfig(**saved["config"]))
        dimensions = config["observation"]
        if (model.config.anchor_dim, model.config.specialist_dim) != (dimensions["anchor_dim"], dimensions["specialist_dim"]):
            raise ValueError("Controller hidden dimensions differ from configured backbones")
        model.load_state_dict(saved["model_state"])
        model.to(config.get("controller_device", "cpu"))
        return NeuralPolicy(runtime, model, overhead)

    def bootstrap(state, legal):
        selected = None
        if state.help:
            selected = next((a for a in legal if a.kind == Kind.SUBMIT), None)
            selected = selected or next((a for a in legal if a.kind == Kind.ABORT), None)
        elif state.help_count == 0:
            selected = next((a for a in legal if a.kind == Kind.HELP and a.target.startswith("s") and
                             a.width == "narrow" and a.quota == "short"), None)
        for kind in (Kind.FINISH, Kind.CONTINUE, Kind.REPAIR):
            selected = selected or next((a for a in legal if a.kind == kind), None)
        if selected is None:
            selected = legal[0]
        return Decision(selected, *overhead()) if overhead else Decision(selected)
    return bootstrap
