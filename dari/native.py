"""Native inference and external collaboration decisions."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import re
import time

from .backends import GenerationRequest
from .pilot_evaluation import final_answer
from .protocol_json import repair_json
from .schema import SpecialistRecord


@dataclass(frozen=True)
class NativeConfig:
    checkpoint_interval: int = 256
    boundary_wait: int = 64
    supporting_tokens: int = 64
    anchor_budget: int = 4096
    specialist_budget: int = 1536
    stage_budget: int = 512
    draft_budget: int = 384
    stages: int = 3
    delegation_mode: str = "frontier"
    anchor_model: str = "A"
    specialist_model: str = "S"

    def __post_init__(self):
        if self.delegation_mode not in {"frontier", "native_intent", "controller_guided"}:
            raise ValueError("Unknown delegation mode")
        if min(self.checkpoint_interval, self.anchor_budget, self.stage_budget,
               self.draft_budget, self.stages) <= 0 or min(
                   self.boundary_wait, self.supporting_tokens, self.specialist_budget) < 0:
            raise ValueError("Invalid native token limits")


@dataclass(frozen=True)
class NativeAction:
    kind: str
    target: str = ""
    mode: str = "auto"


@dataclass(frozen=True)
class ReasoningObject:
    object_id: str
    source_token_span: tuple[int, int]
    original_content: str
    completion_status: str = "complete"
    boundary_status: str = "safe"
    availability: bool = True
    hidden: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class AssistanceTask:
    delegation_mode: str
    target: ReasoningObject
    assistance_mode: str
    supporting_context: str
    supporting_span: tuple[int, int]
    local_budget: int
    native_intent: str | None = None
    fallback_reason: str | None = None


@dataclass
class Session:
    task: AssistanceTask
    history: list[dict] = field(default_factory=list)
    draft: str | None = None
    stages: int = 0
    features: dict = field(default_factory=dict)


@dataclass
class NativeState:
    task: object
    prompt_ids: list[int]
    context_ids: list[int]
    seed: int
    generated_ids: list[int] = field(default_factory=list)
    reasoning_spans: list[tuple[int, int]] = field(default_factory=list)
    injections: list[dict] = field(default_factory=list)
    checkpoints: list[ReasoningObject] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    observations: list[int] = field(default_factory=list)
    candidate: str | None = None
    termination: str | None = None
    session: Session | None = None
    specialist_tokens: int = 0
    boundary: ReasoningObject | None = None
    rng_state: object = None
    anchor_hidden: list[float] = field(default_factory=list)

    @property
    def position(self):
        return len(self.context_ids)

    @property
    def attention_mask(self):
        return [1] * self.position

    @property
    def position_ids(self):
        return list(range(self.position))


class NativeRuntime:
    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config or NativeConfig()
        if not all(callable(getattr(backend, name, None)) for name in (
            "native_prompt_ids", "native_encode", "native_decode", "native_generate", "native_limit"
        )):
            raise ValueError("Backend must support native token inference")

    def initial(self, task, seed=0):
        prompt = ("Solve the problem step by step using the provided evidence and constraints. "
                  "End with a separate line FINAL: followed by only the final answer.\n"
                  + json.dumps({"task": task.question, "sources": task.sources}, ensure_ascii=False))
        ids = self.backend.native_prompt_ids(self.config.anchor_model, prompt)
        if not ids or len(ids) >= self.backend.native_limit(self.config.anchor_model):
            raise ValueError("Initial prompt does not fit native context")
        return NativeState(task, list(ids), list(ids), seed)

    def decode(self, ids):
        return self.backend.native_decode(self.config.anchor_model, ids)

    def advance(self, state, count):
        remaining = min(self.config.anchor_budget - len(state.generated_ids),
                        self.backend.native_limit(self.config.anchor_model) - state.position)
        cap = min(count, remaining)
        if cap <= 0:
            state.termination = "budget_or_context"
            return
        start = state.position
        tokens, state.rng_state, eos, hidden, timing = self.backend.native_generate(
            self.config.anchor_model, state.context_ids, cap, state.seed, state.rng_state)
        state.context_ids.extend(tokens)
        state.generated_ids.extend(tokens)
        state.reasoning_spans.append((start, state.position))
        state.anchor_hidden = hidden
        state.events.append({"kind": "anchor", "input_tokens": start, "output_tokens": len(tokens),
                             "prefill_kind": "initial" if start == len(state.prompt_ids) else "replay", **timing})
        text = self.current_text(state)
        marker = re.search(r"(?im)^FINAL:[ \t]*(\S[^\n]*)\n", text)
        state.candidate = marker.group(1).strip() if marker else None
        if eos or len(tokens) < cap or cap == remaining:
            state.termination = "eos" if eos else "budget_or_context"
            state.candidate = final_answer(text) if text.strip() else None

    def segment_start(self, state):
        return state.injections[-1]["span"][1] if state.injections else len(state.prompt_ids)

    def current_text(self, state):
        return self.decode(state.context_ids[self.segment_start(state):])

    def frontier(self, state):
        start = self.segment_start(state)
        if state.checkpoints:
            start = max(start, state.checkpoints[-1].source_token_span[1])
        text = self.decode(state.context_ids[start:])
        segment = self.current_text(state)
        if segment.count("```") % 2 or segment.count("$$") % 2:
            return None
        if not text.strip() or not re.search(r"(?:\n\s*\n|[.!?][ \t]*\n|```[ \t]*\n)$", text):
            return None
        ids = state.context_ids[start:]
        for offset in range(1, len(ids)):
            if re.search(r"\n\s*\n$", self.decode(ids[:offset])):
                suffix = self.decode(ids[offset:])
                if suffix.strip():
                    start, text = state.position - len(ids) + offset, suffix
        return ReasoningObject("current_frontier", (start, state.position), text, hidden=state.anchor_hidden)

    def objects(self, state):
        current = state.boundary
        if current is None:
            return {}
        objects = {"current_frontier": current}
        if len(state.checkpoints) >= 2:
            objects["previous_frontier"] = state.checkpoints[-2]
        start, end = current.source_token_span
        if state.candidate is not None and re.search(r"(?im)^FINAL:", current.original_content):
            objects["current_candidate"] = ReasoningObject("current_candidate", (start, end), state.candidate)
        cue = re.compile(r"^(?:next, we need to|the remaining issue is|we need to verify whether|it is unclear how to)\b.+[.?]$", re.I)
        for pos in range(start, end):
            suffix = self.decode(state.context_ids[pos:end]).strip()
            if cue.fullmatch(suffix) and not re.search(r"\b(?:already|no longer|need not|do not|not need)\b", suffix, re.I):
                objects["native_open_issue"] = ReasoningObject("native_open_issue", (pos, end), suffix,
                                                               hidden=current.hidden)
                break
        return objects

    def assistance(self, state, action):
        objects = self.objects(state)
        delegation = self.config.delegation_mode
        target = action.target if delegation == "controller_guided" else "current_frontier"
        issue = objects.get("native_open_issue")
        fallback = None
        if delegation == "native_intent":
            if issue:
                target = "native_open_issue"
            else:
                fallback = "no_current_explicit_open_issue"
        obj = objects[target]
        start, _ = obj.source_token_span
        supporting = max(self.segment_start(state), start - self.config.supporting_tokens)
        supporting = min(start, supporting)
        if action.mode == "explore":
            supporting = start
        return AssistanceTask(delegation, obj, action.mode, self.decode(state.context_ids[supporting:start]),
                              (supporting, start), self.config.stage_budget,
                              issue.original_content if issue and target == "native_open_issue" else None,
                              fallback)

    def specialist_request(self, state, session):
        assistance = asdict(session.task)
        del assistance["target"]["hidden"]
        packet = {"question": state.task.question, "sources": state.task.sources,
                  "assistance": assistance, "private_history": session.history}
        directions = {"auto": "Choose the most useful local goal.", "advance": "Advance the unresolved local work.",
                      "verify": "Verify the selected reasoning for concrete errors.",
                      "explore": "Find an independent alternative from the stated premises."}
        prompt = ("Identify and solve a useful local problem in the selected reasoning object. "
                  + directions[session.task.assistance_mode] + " "
                  "Respect the selected assistance direction. State missing prerequisites where needed. "
                  "Return exactly two JSON strings, work and draft.\n" + json.dumps(packet, ensure_ascii=False))
        cap = min(self.config.stage_budget, self.config.specialist_budget - state.specialist_tokens)
        return GenerationRequest(self.config.specialist_model, "specialist", prompt, cap,
                                 sum(event["kind"] == "specialist" for event in state.events), seed=state.seed)

    def specialist_fits(self, state, session):
        request = self.specialist_request(state, session)
        return request.max_new_tokens > 0 and self.backend.count_tokens(request.model_id, request.prompt) + (
            request.max_new_tokens) <= self.backend.native_limit(request.model_id)

    def injection_ids(self, action, text):
        payload = json.dumps(text, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
        return self.backend.native_encode(self.config.anchor_model,
            "\n\n[" + action + "]\n" + payload + "\n[End intervention]\n\n")

    def actions(self, state):
        if state.termination:
            return []
        if state.session:
            result = [NativeAction("abort")]
            if state.session.draft is not None and state.position + len(
                self.injection_ids("External assistance", state.session.draft)
            ) < self.backend.native_limit(self.config.anchor_model):
                result.append(NativeAction("submit"))
            if state.session.stages < self.config.stages and self.specialist_fits(state, state.session):
                result.append(NativeAction("expand"))
            return result
        result = [NativeAction("continue")]
        if state.candidate is not None:
            result.append(NativeAction("finish"))
        if state.boundary:
            if state.position + len(self.injection_ids("Self repair", self.repair_instruction())) < self.backend.native_limit(self.config.anchor_model):
                result.append(NativeAction("self_repair"))
            options = [NativeAction("help")]
            if self.config.delegation_mode == "controller_guided":
                options = [NativeAction("help", target, mode) for target in self.objects(state)
                           for mode in ("auto", "advance", "verify", "explore")
                           if mode != "explore" or (
                               target == "native_open_issue" and
                               not re.search(r"\b(?:this|that|above|previous|it)\b",
                                             self.objects(state)[target].original_content, re.I))]
            result.extend(action for action in options if self.specialist_fits(state, Session(self.assistance(state, action))))
        return result

    @staticmethod
    def repair_instruction():
        return "Check the current reasoning for a concrete error or omission, then continue the solution."

    def inject(self, state, kind, text):
        ids = self.injection_ids(kind, text)
        start = state.position
        state.context_ids.extend(ids)
        state.injections.append({"kind": kind, "content": text, "span": (start, state.position), "token_ids": ids})
        state.boundary = None
        state.candidate = None

    def step(self, state, action):
        if action not in self.actions(state):
            raise ValueError(f"Illegal native action: {action}")
        state = deepcopy(state)
        if action.kind == "continue":
            state.boundary = None
            self.advance(state, self.config.checkpoint_interval)
            state.observations.append(state.position)
            for _ in range(self.config.boundary_wait + 1):
                if state.termination:
                    break
                state.boundary = self.frontier(state)
                if state.boundary:
                    state.checkpoints.append(state.boundary)
                    break
                if _ < self.config.boundary_wait:
                    self.advance(state, 1)
        elif action.kind == "finish":
            state.termination = "finish"
        elif action.kind == "abort":
            state.session = None
            state.boundary = None
        elif action.kind == "submit":
            self.inject(state, "External assistance", state.session.draft)
            state.session = None
        elif action.kind == "self_repair":
            self.inject(state, "Self repair", self.repair_instruction())
        elif action.kind in {"help", "expand"}:
            if action.kind == "help":
                state.session = Session(self.assistance(state, action))
            session = state.session
            request = self.specialist_request(state, session)
            generation = self.backend.generate(request)
            state.specialist_tokens += generation.usage.output_tokens
            session.stages += 1
            error = None
            try:
                repaired, _ = repair_json(generation.text)
                record = SpecialistRecord.parse(repaired)
                if self.backend.count_text_tokens(request.model_id, record.draft.text) > self.config.draft_budget:
                    raise ValueError("Draft exceeds token quota")
                session.history.append({"work": record.work, "draft": record.draft.text})
                session.draft = record.draft.text
                session.features = generation.features
            except ValueError as exc:
                error = str(exc)
            state.events.append({"kind": "specialist", "prompt": request.prompt, "output": generation.text,
                                 "usage": asdict(generation.usage), "ms": generation.latency_ms, "error": error})
        return state

    def run(self, task, policy=None, seed=0):
        state = self.initial(task, seed)
        while not state.termination:
            started = time.perf_counter()
            legal = self.actions(state)
            observed = time.perf_counter()
            action = policy(deepcopy(state), legal[:]) if policy else NativeAction("continue")
            state.events.append({"kind": "decision", "action": asdict(action),
                                 "state_ms": (observed-started)*1000,
                                 "controller_ms": (time.perf_counter()-observed)*1000})
            state = self.step(state, action)
        return state

    def standard(self, task, seed=0):
        return self.run(task, seed=seed)
