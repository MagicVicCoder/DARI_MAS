"""Replayable runtime records. Gold labels intentionally live elsewhere."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any


def object_json(text: str, fields: set[str]) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError("Output must be one complete JSON object") from exc
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ValueError(f"Expected fields: {sorted(fields)}")
    return obj


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HelpRequest:
    instruction: str
    target: str
    required_refs: list[str] = field(default_factory=list)
    fallback: bool = False
    reason: str | None = None

    @classmethod
    def parse(cls, obj):
        if not isinstance(obj, dict) or set(obj) != {"instruction", "target", "required_refs"}:
            raise ValueError("Invalid help_request fields")
        if any(not isinstance(obj[key], str) or not obj[key].strip() for key in ("instruction", "target")):
            raise ValueError("Help instruction and target must be nonempty text")
        if not isinstance(obj["required_refs"], list) or any(not isinstance(ref, str) for ref in obj["required_refs"]):
            raise ValueError("required_refs must be source IDs")
        return cls(**obj)


@dataclass(frozen=True)
class AnchorRecord:
    step: str
    refs: list[str]
    goal: str | None
    answer: str | None
    help_request: HelpRequest | None = None
    invalidate_answer: bool = False

    @classmethod
    def parse(cls, text: str) -> AnchorRecord:
        try:
            obj = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ValueError("Anchor output must be a JSON object") from exc
        if not isinstance(obj, dict) or not {"step", "refs", "goal", "answer"} <= set(obj) or set(obj)-{"step", "refs", "goal", "answer", "help_request", "invalidate_answer"}:
            raise ValueError("Invalid Anchor record fields")
        if obj.get("help_request") is not None:
            obj["help_request"] = HelpRequest.parse(obj["help_request"])
        if type(obj.get("invalidate_answer", False)) is not bool:
            raise ValueError("invalidate_answer must be boolean")
        if not isinstance(obj["step"], str) or not obj["step"].strip():
            raise ValueError("Anchor step must be nonempty text")
        refs = obj["refs"]
        if not isinstance(refs, list) or len(refs) > 4 or any(not isinstance(x, str) for x in refs):
            raise ValueError("refs must contain at most four source IDs")
        for name in ("goal", "answer"):
            if obj[name] is not None and (not isinstance(obj[name], str) or not obj[name].strip()):
                raise ValueError(f"{name} must be nonempty text or null")
        return cls(**obj)


@dataclass(frozen=True)
class Draft:
    text: str

    @classmethod
    def parse(cls, text: str) -> Draft:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Draft must be nonempty assistance text")
        return cls(text)


@dataclass(frozen=True)
class SpecialistRecord:
    work: str
    draft: Draft

    @classmethod
    def parse(cls, text: str) -> SpecialistRecord:
        obj = object_json(text, {"work", "draft"})
        if not isinstance(obj["work"], str) or not obj["work"].strip():
            raise ValueError("Private work must be nonempty text")
        if not isinstance(obj["draft"], str):
            raise ValueError("Specialist schema v3 requires a string draft; legacy objects are incompatible")
        return cls(obj["work"], Draft.parse(obj["draft"]))


def legacy_specialist_output(text: str) -> dict:
    """Read-only historical analysis, never an alternate online generation parser."""
    obj = object_json(text, {"work", "draft"})
    required = {"target", "finding", "support", "change", "unresolved"}
    if not isinstance(obj["work"], str) or not isinstance(obj["draft"], dict) or set(obj["draft"]) != required:
        raise ValueError("Invalid historical Specialist output")
    if any(not isinstance(value, str) for value in obj["draft"].values()):
        raise ValueError("Historical draft fields must be text")
    return obj


class Kind(str, Enum):
    CONTINUE = "continue"
    REPAIR = "self_repair"
    HELP = "help"
    FINISH = "finish"
    SUBMIT = "submit"
    ABORT = "abort"
    EXPAND = "expand"


@dataclass(frozen=True)
class Action:
    kind: Kind
    target: str = ""
    template: str = ""
    width: str = ""
    quota: str = ""

    @property
    def key(self) -> str:
        return "|".join((self.kind.value, self.target, self.template, self.width, self.quota))


@dataclass(frozen=True)
class Step:
    step_id: str
    record: AnchorRecord


@dataclass
class HelpSession:
    help_id: str
    action: Action
    private_outputs: list[str] = field(default_factory=list)
    draft: Draft | None = None
    stage: int = 0


@dataclass(frozen=True)
class Feature:
    source: str
    values: list[float]


@dataclass(frozen=True)
class CallEvent:
    role: str
    prompt: str
    output: str
    valid: bool
    error: str | None = None
    usage: dict | None = None
    latency_ms: float | None = None
    raw_output: str | None = None

