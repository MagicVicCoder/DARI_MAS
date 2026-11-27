"""Frozen exploratory answer rules shared by all Phase 2A conditions."""
from collections import Counter
from fractions import Fraction
import re
import string
import json
import ast

EVALUATOR_VERSION = "phase2a-eval-v3-rational-arithmetic"


def final_answer(text):
    if text is None:
        return None
    # A one-value JSON wrapper is unambiguous. Never pick a value from a
    # multi-field answer or search a reasoning paragraph for a convenient number.
    for _ in range(3):
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            break
        if isinstance(obj, dict) and len(obj)==1:
            obj=next(iter(obj.values()))
        if isinstance(obj,(str,int,float)) and not isinstance(obj,bool):
            text=str(obj)
        else:
            break
    # Same extraction for standard prose and structured candidate strings.
    marker = text.rfind("\\boxed{")
    if marker >= 0:
        start, depth = marker+7, 1
        for position in range(start, len(text)):
            depth += (text[position] == "{") - (text[position] == "}")
            if depth == 0:
                return text[start:position].strip()
    lines = re.findall(r"(?:^|\n)\s*(?:FINAL(?: ANSWER)?|####)\s*:?\s*(.+)", text, re.I)
    if lines:
        return lines[-1].strip()
    marked = re.findall(r"(?:final answer|the answer)\s*(?:is|:)\s*(.+)", text, re.I)
    return (marked[-1] if marked else text).strip()


def numeric(text):
    text = final_answer(text)
    if text is None:
        return None
    text = text.replace(",", "").replace("−", "-").replace("\\,", "").strip().strip("$. ")
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    match = re.fullmatch(r"([\d.\s+*/()^\-]+)(?:\s*(?:[A-Za-z][A-Za-z\s-]*|%))?", text)
    if not match:
        return None
    try:
        expression=match[1].strip().replace("^","**")
        if len(expression)>256:
            return None
        def visit(node):
            if isinstance(node,ast.Constant) and type(node.value) in (int,float):
                return Fraction(ast.get_source_segment(expression,node))
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                value=visit(node.operand)
                return value if isinstance(node.op,ast.UAdd) else -value
            if isinstance(node,ast.BinOp):
                left,right=visit(node.left),visit(node.right)
                if isinstance(node.op,ast.Add): return left+right
                if isinstance(node.op,ast.Sub): return left-right
                if isinstance(node.op,ast.Mult): return left*right
                if isinstance(node.op,ast.Div): return left/right
                if isinstance(node.op,ast.Pow) and right.denominator==1 and abs(right)<=12: return left**int(right)
            raise ValueError("Not a bounded rational arithmetic expression")
        return visit(ast.parse(expression,mode="eval").body)
    except (ValueError, ZeroDivisionError, SyntaxError, OverflowError):
        return None


def normalized(text):
    text = "".join(character for character in (text or "").lower() if character not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(dataset, answer, gold):
    extracted = final_answer(answer)
    if dataset == "hotpotqa":
        left, right = normalized(extracted), normalized(gold)
        em = float(left == right)
        if left in {"yes", "no", "noanswer"} or right in {"yes", "no", "noanswer"}:
            f1 = em
        else:
            a, b = left.split(), right.split()
            common = sum((Counter(a) & Counter(b)).values())
            f1 = 2*common/(len(a)+len(b)) if a and b else float(a == b)
        return {"reward": em, "f1": f1, "extracted": extracted, "evaluator_status": "scored"}
    if dataset == "gsm8k":
        left, right = numeric(extracted), numeric(gold)
        return {"reward": float(left is not None and right is not None and left == right),
                "f1": None, "extracted": extracted,
                "evaluator_status": "scored" if left is not None else "unparsed_answer"}
    if dataset != "math500":
        raise ValueError("Unsupported dataset")
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    if extracted is None:
        return {"reward": 0., "f1": None, "extracted": None, "evaluator_status": "no_answer"}
    configs = (LatexExtractionConfig(), ExprExtractionConfig())
    # Wrapping concise candidate expressions supplies a uniform math delimiter.
    expected = parse("$"+gold.strip("$")+"$", extraction_config=configs)
    actual = parse("$"+extracted.strip("$")+"$", extraction_config=configs)
    symbolic = lambda values: any(not isinstance(value, str) for value in values)
    uncertain = not symbolic(expected) or not symbolic(actual)
    return {"reward": float(verify(expected, actual)), "f1": None, "extracted": extracted,
            "evaluator_status": "manual_review_required" if uncertain else "scored"}
