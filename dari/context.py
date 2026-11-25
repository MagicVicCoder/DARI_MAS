"""Explicit source selection and prompt assembly; no generative summarizer."""
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import EpisodeState


@dataclass(frozen=True)
class ContextPack:
    target: str
    span_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class PromptPacket:
    text: str
    feature_spans: dict[str, tuple[int, int]]


def encode_with_spans(body, tracked: dict[tuple, str], prefix: str = "") -> PromptPacket:
    """Serialize JSON and preserve exact ID-specific escaped value ranges.

    Matching a text substring after serialization is ambiguous when source text
    repeats or equals an instruction/key. Paths identify the intended values.
    """
    chunks, ranges = [prefix], {}
    length = len(prefix)

    def append(text):
        nonlocal length
        chunks.append(text)
        length += len(text)

    def render(value, path):
        if isinstance(value, dict):
            append("{")
            for i, (key, child) in enumerate(value.items()):
                if i:
                    append(", ")
                append(json.dumps(key, ensure_ascii=False)+": ")
                render(child, path+(key,))
            append("}")
        elif isinstance(value, list):
            append("[")
            for i, child in enumerate(value):
                if i:
                    append(", ")
                render(child, path+(i,))
            append("]")
        else:
            encoded = json.dumps(value, ensure_ascii=False)
            if path in tracked and isinstance(value, str) and value:
                ranges[tracked[path]] = (length+1, length+len(encoded)-1)
            append(encoded)
    render(body, ())
    return PromptPacket("".join(chunks), ranges)


def spans(state: "EpisodeState") -> dict[str, str]:
    result = dict(state.task.sources)
    result.update(state.artifacts)
    for step in state.steps:
        result[step.step_id] = step.record.step
        if step.record.goal:
            result["g" + step.step_id[1:]] = step.record.goal
    return result


def build_pack(state: "EpisodeState", target: str, width: str) -> ContextPack:
    if width not in {"narrow", "wide"}:
        raise ValueError("Unknown context width")
    available = spans(state)
    if target not in available:
        raise ValueError("Unknown help target")
    chosen = [target]
    request = getattr(state, "help_request", None)
    if request:
        chosen.extend(request.required_refs)
    owner_id = "s" + target[1:] if target.startswith("g") else target
    owners = {s.step_id: s for s in state.steps}
    if owner_id in owners:
        chosen.append(owner_id)
        chosen.extend(owners[owner_id].record.refs)
    if width == "wide":
        for step in state.steps[-getattr(state, "wide_history", 4):]:
            chosen.append(step.step_id)
            chosen.extend(step.record.refs)
    ids = tuple(dict.fromkeys(i for i in chosen if i in available))
    text = json.dumps({"task": state.task.question, "target": target,
                       "context": {i: available[i] for i in ids}}, ensure_ascii=False)
    return ContextPack(target, ids, text)


def anchor_input(state: "EpisodeState", instruction: str, temporary_help: tuple[str, str] | None = None) -> PromptPacket:
    history = []
    for step in state.steps:
        record = {"step": step.record.step, "refs": step.record.refs}
        history.append({"id": step.step_id, **record})
    body = {"task": state.task.question, "sources": state.task.sources,
            "steps": history,
            "evidence": dict(state.artifacts), "candidate": state.candidate}
    if temporary_help is not None:
        body["temporary_help"] = {temporary_help[0]: temporary_help[1]}
    rule = ("Return one JSON object: step (new reasoning text), refs (at most 4 existing IDs), "
            "goal (active subgoal or null), answer (complete answer as a JSON string or null, never a number or boolean), "
            + ("help_request (null or {instruction: an open-ended local task, target: current_step or an existing ID, required_refs: existing IDs}), " if state.generate_help_request else "") +
            "invalidate_answer (boolean; true revokes an obsolete candidate before applying answer). "
            "Make substantive new progress each step; do not merely restate plans or repeat calculations. "
            "When solved, set answer to the concise final result, not an explanation; use standard LaTeX for mathematical expressions. "
            "The answer string must contain the final value itself, not a JSON object or an encoded JSON document. "
            "Set answer ONLY when the ORIGINAL QUESTION is fully solved; intermediate results belong in step with answer=null. "
            "Do not emit id or any other extra fields: history IDs are assigned by Runtime, not by you. "
            "Use refs=[] if citing no provided ID; never invent numeric references such as '1'. "
            "If unfinished, solve a concrete part now. Use only existing source IDs. "
            + ("Request help only for a genuinely unresolved task; do not repeat an already completed request. " if state.generate_help_request else "") +
            "External help is fallible. Reference its ID only if retaining its evidence. ")
    rule += ("Reason naturally about the ORIGINAL QUESTION. Each checkpoint may contain several connected deductions; "
             "actually calculate or infer a result, rather than outputting a plan. Check previous conclusions against the original evidence. "
             "If solved, answer the complete original question now; do not leave a solved result only in step. "
             "Use plain notation or properly double-escaped JSON backslashes. ")
    if state.generate_help_request:
        rule += ("If unfinished, propose a concrete unresolved local reasoning task in help_request whenever one can be identified. "
                 "This is only a proposal; Controller decides whether any assistance is invoked. "
                 "A proposal does not require external help. goal is the unresolved question; instruction specifies useful work to solve it. "
                 "Avoid generic checking if a specific calculation, relation, alternative or evidence issue exists. "
                 "Use null when solved or when no meaningful local task exists. Never request a task already completed in history. ")
    tracked = {("task",): "task", ("candidate",): "candidate"}
    tracked.update({("sources", key): key for key in state.task.sources})
    tracked.update({("evidence", key): key for key in state.artifacts})
    for i, step in enumerate(state.steps):
        tracked[("steps", i, "step")] = step.step_id

    return encode_with_spans(body, tracked, rule+instruction+"\n")


def anchor_prompt(state: "EpisodeState", instruction: str, temporary_help: tuple[str, str] | None = None) -> str:
    return anchor_input(state, instruction, temporary_help).text
