"""Generation boundary. ScriptedBackend is a synthetic fixture, never a benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Protocol

from .costs import Usage
from .schema import Feature


@dataclass(frozen=True)
class GenerationRequest:
    model_id: str
    role: str
    prompt: str
    max_new_tokens: int
    call_index: int
    feature_texts: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    feature_spans: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Generation:
    text: str
    usage: Usage
    features: dict[str, Feature] = field(default_factory=dict)


class Backend(Protocol):
    synthetic: bool
    deployment_identity: dict
    def count_tokens(self, model_id: str, text: str) -> int: ...
    def count_text_tokens(self, model_id: str, text: str) -> int: ...
    def generate(self, request: GenerationRequest) -> Generation: ...


class ScriptedBackend:
    """Stateless replayable fixture with explicitly fake tokens/hidden features.

    A real adapter must tokenize its exact formatted model input; report all
    usage, including failed calls; pool actual forward activations for the named
    spans and generated output; and preserve seed/runtime state on replay.
    """
    synthetic = True

    def __init__(self, outputs: dict[str, list[str]], dimensions: dict[str, int] | None = None):
        self.outputs = {k: tuple(v) for k, v in outputs.items()}
        self.dimensions = dimensions or {"anchor": 12, "specialist": 9}

    def count_tokens(self, model_id: str, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def count_text_tokens(self, model_id: str, text: str) -> int:
        return self.count_tokens(model_id, text)

    @property
    def deployment_identity(self) -> dict:
        return {"engine": type(self).__module__+":"+type(self).__qualname__,
                "representation": "synthetic-sha256-v1", "hidden_dims": self.dimensions,
                "outputs": hashlib.sha256(json.dumps(self.outputs, sort_keys=True).encode()).hexdigest(),
                "adapters": {}}

    def _feature(self, role: str, text: str) -> Feature:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[i % len(digest)] / 127.5 - 1 for i in range(self.dimensions[role])]
        return Feature(role, values)

    def generate(self, request: GenerationRequest) -> Generation:
        try:
            text = self.outputs[request.role][request.call_index]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"No scripted {request.role} output #{request.call_index}") from exc
        features = {k: self._feature(request.role, v) for k, v in request.feature_texts.items()}
        features["output"] = self._feature(request.role, text)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "draft" in obj:
                features["draft"] = self._feature(request.role, json.dumps(obj["draft"], sort_keys=True))
        except ValueError:
            pass
        return Generation(text, Usage(request.model_id,
            self.count_tokens(request.model_id, request.prompt),
            self.count_tokens(request.model_id, text)), features)
