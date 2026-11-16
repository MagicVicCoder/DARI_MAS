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
class AnchorRecord:
    step: str
    refs: list[str]
    goal: str | None
    answer: str | None

    @classmethod
    def parse(cls, text: str) -> AnchorRecord:
        obj = object_json(text, {"step", "refs", "goal", "answer"})
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
    target: str
    finding: str
    support: str
    change: str
    unresolved: str

    @classmethod
    def parse(cls, text: str) -> Draft:
        obj = object_json(text, {"target", "finding", "support", "change", "unresolved"})
        if any(not isinstance(v, str) for v in obj.values()):
            raise ValueError("All draft fields must be text")
        if any(not obj[k].strip() for k in ("target", "finding", "support", "change")):
            raise ValueError("Draft requires target, finding, support and change")
        return cls(**obj)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class SpecialistRecord:
    work: str
    draft: Draft

    @classmethod
    def parse(cls, text: str) -> SpecialistRecord:
        obj = object_json(text, {"work", "draft"})
        if not isinstance(obj["work"], str):
            raise ValueError("Private work must be text")
        return cls(obj["work"], Draft.parse(json.dumps(obj["draft"], ensure_ascii=False)))


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

