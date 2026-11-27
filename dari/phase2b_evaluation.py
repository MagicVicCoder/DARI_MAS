"""Uniform corrected-evaluator sensitivity; historical scores remain untouched."""
import re

from .pilot_evaluation import final_answer, score as historical_score

EVALUATOR_VERSION = "phase2b-v1-single-final-math-span"


def math_expression(answer):
    extracted = final_answer(answer)
    if extracted is None:
        return None
    spans = re.findall(r"(?<!\\)\$([^$]+)(?<!\\)\$", extracted)
    if len(spans) == 1:
        outside = re.sub(r"(?<!\\)\$[^$]+(?<!\\)\$", "", extracted)
        if not re.search(r"\d", outside):
            return spans[0].strip()
    match = re.fullmatch(r"(?:the\s+)?(?:simplified\s+)?(?:expression|result|answer|probability|value)\s+is\s*:?\s*(.+?)\.?", extracted, re.I)
    return match[1].strip() if match else extracted


def score(dataset, answer, gold):
    if dataset != "math500":
        return historical_score(dataset, answer, gold)
    return historical_score(dataset, math_expression(answer), gold)
