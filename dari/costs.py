"""Unit-safe, heterogeneous inference accounting; token sums are diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
import math


def nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class Usage:
    model_id: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    context_tokens: int = 0
    observed_cost: float | None = None
    observed_unit: str | None = None

    def __post_init__(self):
        for name in ("input_tokens", "output_tokens", "cached_input_tokens", "context_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("Cached input is a subset of total input")
        if self.observed_cost is not None:
            nonnegative(self.observed_cost, "observed_cost")
            if not self.observed_unit:
                raise ValueError("Observed cost requires an explicit unit")


@dataclass(frozen=True)
class ModelProfile:
    """Coefficients must be calibrated for one deployment, NOT inferred from size.

    Linear API prices are exact for their declared tariff. Local resource profiles
    may add empirically fitted prefill-square and decode-context terms. A real
    backend should supply measured cost when available. Provenance is mandatory.
    """
    model_id: str
    unit: str
    uncached_input: float
    output: float
    cached_input: float = 0.0
    prefill_square: float = 0.0
    decode_context: float = 0.0
    call_overhead: float = 0.0
    provenance: str = ""
    synthetic: bool = False

    def __post_init__(self):
        if not self.model_id or not self.unit or not self.provenance:
            raise ValueError("Profile needs model ID, unit, and calibration/pricing provenance")
        for name in ("uncached_input", "output", "cached_input", "prefill_square", "decode_context", "call_overhead"):
            nonnegative(getattr(self, name), name)

    def estimate(self, usage: Usage) -> float:
        if usage.model_id != self.model_id:
            raise ValueError("Usage/profile model mismatch")
        uncached = usage.input_tokens - usage.cached_input_tokens
        length = max(usage.context_tokens, usage.input_tokens)
        attend_pairs = length * usage.output_tokens + usage.output_tokens * (usage.output_tokens - 1) / 2
        return (self.call_overhead + self.uncached_input * uncached
                + self.cached_input * usage.cached_input_tokens + self.output * usage.output_tokens
                + self.prefill_square * uncached * uncached + self.decode_context * attend_pairs)


@dataclass
class CostLedger:
    profiles: dict[str, ModelProfile]
    entries: list[tuple[Usage, float]] = field(default_factory=list)
    overheads: list[tuple[str, float]] = field(default_factory=list)
    unmeasured_overheads: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.profiles:
            raise ValueError("At least one explicit cost profile is required")
        if any(k != v.model_id for k, v in self.profiles.items()):
            raise ValueError("Profile registry key must match model ID")
        if len({p.unit for p in self.profiles.values()}) != 1:
            raise ValueError("Cannot combine incompatible cost units")
        if len({p.synthetic for p in self.profiles.values()}) != 1:
            raise ValueError("Synthetic and measured profiles must not be mixed")

    @property
    def unit(self) -> str:
        return next(iter(self.profiles.values())).unit

    @property
    def total(self) -> float:
        return sum(c for _, c in self.entries) + sum(c for _, c in self.overheads)

    def charge(self, usage: Usage) -> float:
        profile = self.profiles[usage.model_id]
        if usage.observed_cost is None:
            cost = profile.estimate(usage)
        else:
            if usage.observed_unit != self.unit:
                raise ValueError("Observed cost has incompatible units")
            cost = usage.observed_cost
        self.entries.append((usage, cost))
        return cost

    def charge_overhead(self, name: str, cost: float, unit: str) -> None:
        nonnegative(cost, "overhead")
        if unit != self.unit:
            raise ValueError("Overhead has incompatible units")
        self.overheads.append((name, cost))

    def tokens_by_model(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for usage, _ in self.entries:
            row = result.setdefault(usage.model_id, dict(input=0, output=0, cached_input=0, calls=0))
            row["input"] += usage.input_tokens
            row["output"] += usage.output_tokens
            row["cached_input"] += usage.cached_input_tokens
            row["calls"] += 1
        return result
