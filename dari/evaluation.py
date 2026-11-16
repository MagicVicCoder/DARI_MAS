"""Offline-only evaluators. Task/runtime observations never contain gold.

For paper benchmarks supply the official normalization/evaluation adapter;
ExactMatch is deliberately not advertised as a MATH or HumanEval evaluator.
"""
from collections import Counter
from dataclasses import dataclass
import re
import string
from typing import Callable, Protocol


class Evaluator(Protocol):
    def score(self, answer: str | None, gold: str) -> float: ...


class ExactMatch:
    def score(self, answer: str | None, gold: str) -> float:
        return float(answer is not None and answer.strip() == gold.strip())


@dataclass(frozen=True)
class ChoiceAccuracy:
    choices: str = "ABCDE"

    def score(self, answer: str | None, gold: str) -> float:
        if answer is None:
            return 0.
        answer = answer.strip().upper()
        return float(len(answer) == 1 and answer in self.choices and answer == gold.strip().upper())


def _tokens(text: str) -> list[str]:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


class AnswerF1:
    def score(self, answer: str | None, gold: str) -> float:
        if answer is None:
            return 0.
        a, g = _tokens(answer), _tokens(gold)
        if not a or not g:
            return float(a == g)
        shared = sum((Counter(a) & Counter(g)).values())
        return 2. * shared / (len(a) + len(g))


@dataclass(frozen=True)
class OfficialEvaluatorAdapter:
    """Inject an official scorer; code execution must be sandboxed externally."""
    scorer: Callable[[str, str], float]

    def score(self, answer: str | None, gold: str) -> float:
        if answer is None:
            return 0.
        value = float(self.scorer(answer, gold))
        if not 0. <= value <= 1.:
            raise ValueError("Official scorer must return a normalized score in [0,1]")
        return value
