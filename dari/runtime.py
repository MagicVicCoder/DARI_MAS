"""Finite, replayable local-help state machine. This module never sees gold."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import json
import math
from typing import Callable

from .backends import Backend, GenerationRequest
from .context import anchor_input, anchor_prompt, build_pack, encode_with_spans, spans
from .costs import CostLedger, ModelProfile, Usage
from .schema import Action, AnchorRecord, CallEvent, Feature, HelpRequest, HelpSession, Kind, SpecialistRecord, Step, Task

CONTINUE_INSTRUCTION = "Continue reasoning."
SUBMIT_INSTRUCTION = "Use the fallible local help if useful, then continue."


@dataclass(frozen=True)
class RuntimeConfig:
    anchor_model: str = "A"
    specialist_model: str = "S"
    budget: float = 100000.0
    reference_cost: float = 1000.0
    anchor_output: int = 256
    oneshot_output: int = 480
    draft_limit: int = 384
    specialist_input_limit: int = 8192
    max_anchor_steps: int = 16
    max_decisions: int = 128
    max_specialist_stages: int = 3
    stage_output_budgets: tuple[int, ...] = (512, 512, 512)
    wide_history: int = 4
    controller_history: int = 8
    experiment_mode: str = "synthetic"
    generate_help_request: bool = True
    repair_protocol_json: bool = False

    def __post_init__(self):
        if self.experiment_mode not in {"synthetic", "real_debug", "formal"}:
            raise ValueError("Invalid experiment mode")
        if self.max_specialist_stages < 1 or len(self.stage_output_budgets) < self.max_specialist_stages or any(type(n) is not int or n <= 0 for n in self.stage_output_budgets):
            raise ValueError("Each specialist stage requires a positive output budget")
        if self.wide_history < 1 or self.controller_history < 1:
            raise ValueError("History windows must be positive")
        if not math.isfinite(self.budget) or self.budget < 0:
            raise ValueError("Budget must be finite and nonnegative")
        if not math.isfinite(self.reference_cost) or self.reference_cost <= 0:
            raise ValueError("Reference cost must be positive")
        for name in ("anchor_output", "oneshot_output", "draft_limit",
                     "specialist_input_limit", "max_anchor_steps", "max_decisions"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class EpisodeState:
    task: Task
    ledger: CostLedger
    budget: float
    steps: list[Step] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    candidate: str | None = None
    help: HelpSession | None = None
    calls: dict[str, int] = field(default_factory=lambda: {"anchor": 0, "specialist": 0})
    events: list[CallEvent] = field(default_factory=list)
    features: dict[str, Feature] = field(default_factory=dict)
    help_count: int = 0
    help_request: HelpRequest | None = None
    wide_history: int = 4
    generate_help_request: bool = True
    decision_log: list[dict] = field(default_factory=list)
    assistance_log: list[dict] = field(default_factory=list)
    terminal: bool = False
    termination_reason: str | None = None
    seed: int = 0
    decisions: int = 0

    @property
    def remaining(self) -> float:
        return max(0., self.budget - self.ledger.total)


@dataclass(frozen=True)
class Decision:
    action: Action
    resource_cost: float | None = None
    unit: str | None = None


class Runtime:
    def __init__(self, backend: Backend, profiles: dict[str, ModelProfile], config: RuntimeConfig | None = None):
        self.backend, self.profiles = backend, dict(profiles)
        self.config = config or RuntimeConfig()
        CostLedger(self.profiles)
        if type(getattr(backend, "synthetic", None)) is not bool:
            raise ValueError("Backend must explicitly declare synthetic provenance")
        if not isinstance(getattr(backend, "deployment_identity", None), dict) or not backend.deployment_identity:
            raise ValueError("Backend must declare model/tokenizer/representation deployment identity")
        json.dumps(backend.deployment_identity, sort_keys=True)
        if any(p.synthetic != backend.synthetic for p in self.profiles.values()):
            raise ValueError("Backend and cost profiles have inconsistent synthetic provenance")
        for model in (self.config.anchor_model, self.config.specialist_model):
            if model not in profiles:
                raise ValueError(f"Missing model profile: {model}")

    def initial(self, task: Task, seed: int = 0) -> EpisodeState:
        if any(k == "task" or k.startswith(("s", "g", "h")) and k[1:].isdigit() for k in task.sources):
            raise ValueError("Source IDs conflict with reserved task/step/goal/help IDs")
        state = EpisodeState(task, CostLedger(self.profiles), self.config.budget, seed=seed,
                             wide_history=self.config.wide_history, generate_help_request=self.config.generate_help_request)
        if self.config.experiment_mode == "real_debug":
            state.ledger.unmeasured_overheads.append("uncalibrated_debug_resource_cost")
        self._anchor(state, CONTINUE_INSTRUCTION)
        return state

    def estimate(self, model: str, prompt: str, output_cap: int) -> float:
        count = self.backend.count_tokens(model, prompt)
        return self.profiles[model].estimate(Usage(model, count, output_cap, context_tokens=count))

    def fits_context(self, request: GenerationRequest) -> bool:
        backend = self.backend
        while hasattr(backend, "backend"):
            backend = backend.backend
        handles = getattr(backend, "handles", {})
        handle = handles.get(request.model_id)
        return handle is None or self.backend.count_tokens(request.model_id, request.prompt) + request.max_new_tokens <= handle.context_limit

    def _anchor_reserve(self, state: EpisodeState, help_payload: tuple[str, str] | None = None) -> float:
        instruction = SUBMIT_INSTRUCTION if help_payload else CONTINUE_INSTRUCTION
        prompt = anchor_prompt(state, instruction, help_payload)
        # A conservative estimate is a guard, not a wall-clock guarantee.
        if help_payload is None:
            prompt += " reserve" * self.config.draft_limit
        return self.estimate(self.config.anchor_model, prompt, self.config.anchor_output)

    def _specialist_request(self, state: EpisodeState) -> GenerationRequest:
        session = state.help
        if session is None:
            raise ValueError("No active help session")
        pack = build_pack(state, session.action.target, session.action.width)
        cap = (self.config.oneshot_output if session.action.quota == "oneshot" else
               self.config.stage_output_budgets[min(session.stage, self.config.max_specialist_stages-1)])
        selected = spans(state)
        body = {"dynamic_request": state.help_request.instruction if state.help_request else "Inspect the unresolved reasoning",
                "stage_instruction": "Extend previous local reasoning" if session.stage else "Initialize local reasoning",
                "selected_context_ids": list(pack.span_ids),
                "original_question_and_constraints": state.task.question,
                "relevant_source_evidence": {key:selected[key] for key in pack.span_ids if key in state.task.sources or key in state.artifacts},
                "anchor_reasoning_context": {key:selected[key] for key in pack.span_ids if key not in state.task.sources and key not in state.artifacts},
                "max_total_output_tokens": cap, "max_draft_tokens": self.config.draft_limit,
                "previous_private_outputs": session.private_outputs}
        instruction = ("You are an independent local reasoning agent assisting the Anchor. Solve the dynamic assistance task yourself. "
            "Perform the calculation, derivation, evidence comparison or other requested reasoning; do not merely describe what Anchor should do. "
            "The original question and source evidence are primary; Anchor reasoning is fallible and may be incomplete or wrong. "
            "Read all original constraints before claiming information is missing. Provide concrete results or verified partial progress with uncertainty. "
            "Return exactly one JSON object with two string fields: work and draft. "
            "In work write your actual local solution reasoning, not a label, placeholder or task restatement. "
            "In draft communicate the useful result with its necessary supporting calculation/evidence; avoid repeating work verbatim. "
            "No target/session metadata is generated. Both strings and JSON syntax count toward total output budget; "
            "draft_max_tokens applies only to decoded draft text. Use valid JSON escaping, or plain mathematical notation to avoid invalid backslashes.\n")
        tracked = {("original_question_and_constraints",): "task"}
        tracked.update({("relevant_source_evidence" if key in state.task.sources or key in state.artifacts else
                         "anchor_reasoning_context", key): key for key in pack.span_ids})
        prompt = encode_with_spans(body, tracked, instruction)
        requested = {i: selected[i] for i in pack.span_ids}
        requested["task"] = state.task.question
        return GenerationRequest(self.config.specialist_model, "specialist", prompt.text, cap,
                                 state.calls["specialist"], requested, state.seed, prompt.feature_spans)

    def request_for_action(self, state: EpisodeState, action: Action) -> GenerationRequest | None:
        """Describe the exact first call before executing it; never generate text.

        Shared by cost features and scheduling. Format retries are separate calls.
        """
        if action.kind == Kind.FINISH:
            return None
        if action.kind == Kind.HELP:
            probe = deepcopy(state)
            probe.help = HelpSession(f"h{state.help_count+1}", action)
            return self._specialist_request(probe)
        if action.kind == Kind.EXPAND:
            return self._specialist_request(state)
        payload = None
        instruction = self.repair_instruction(state) if action.kind == Kind.REPAIR else CONTINUE_INSTRUCTION
        if action.kind == Kind.SUBMIT:
            if state.help is None or state.help.draft is None:
                raise ValueError("Submit requires a complete current draft")
            payload = (state.help.help_id, state.help.draft.text)
            instruction = SUBMIT_INSTRUCTION
        prompt = anchor_input(state, instruction, payload)
        texts = {"task": state.task.question, **spans(state)}
        if state.candidate is not None:
            texts["candidate"] = state.candidate
        texts = {key:value for key,value in texts.items() if key in prompt.feature_spans}
        return GenerationRequest(self.config.anchor_model, "anchor", prompt.text, self.config.anchor_output,
                                 state.calls["anchor"], texts, state.seed, prompt.feature_spans)

    @staticmethod
    def repair_instruction(state):
        request = state.help_request
        return "Address this local reasoning task yourself before continuing: " + (request.instruction if request else "Inspect unresolved reasoning")

    def actions(self, state: EpisodeState) -> list[Action]:
        if state.terminal or state.decisions >= self.config.max_decisions:
            return []
        if state.help is not None:
            result = [Action(Kind.ABORT)]
            session = state.help
            if session.draft is not None:
                payload = (session.help_id, session.draft.text)
                if self._anchor_reserve(state, payload) <= state.remaining:
                    result.insert(0, Action(Kind.SUBMIT))
                if session.action.quota == "short" and session.stage < self.config.max_specialist_stages:
                    request = self._specialist_request(state)
                    need = self.estimate(request.model_id, request.prompt, request.max_new_tokens)
                    if (self.fits_context(request) and self.backend.count_tokens(request.model_id, request.prompt) <= self.config.specialist_input_limit
                            and need + self._anchor_reserve(state) <= state.remaining):
                        result.append(Action(Kind.EXPAND))
            return result
        result = []
        if state.candidate is not None:
            result.append(Action(Kind.FINISH))
        if len(state.steps) >= self.config.max_anchor_steps:
            return result
        for kind in (Kind.CONTINUE, Kind.REPAIR):
            request = self.request_for_action(state, Action(kind))
            assert request is not None
            if self.fits_context(request) and self.estimate(request.model_id, request.prompt, request.max_new_tokens) <= state.remaining:
                result.append(Action(kind))
        if not state.steps:
            return result
        if state.help_request is None:
            return result
        target = state.help_request.target
        for width in ("narrow", "wide"):
            for quota in ("short", "oneshot"):
                action = Action(Kind.HELP, target, "", width, quota)
                request = self.request_for_action(state, action)
                cost = self.estimate(request.model_id, request.prompt, request.max_new_tokens)
                if (self.fits_context(request) and self.backend.count_tokens(request.model_id, request.prompt) <= self.config.specialist_input_limit
                        and cost + self._anchor_reserve(state) <= state.remaining):
                    result.append(action)
        return result

    @staticmethod
    def _terminate(state: EpisodeState, reason: str) -> None:
        state.terminal = True
        state.termination_reason = reason

    def _anchor(self, state: EpisodeState, instruction: str,
                payload: tuple[str, str] | None = None, payload_feature: Feature | None = None) -> None:
        last_error = None
        for attempt in range(2):
            correction = (" Previous output was invalid: " + last_error + ". Return a complete valid JSON record with exactly the specified fields." if last_error else "")
            packet = anchor_input(state, instruction + correction, payload)
            prompt = packet.text
            if len(state.steps) >= self.config.max_anchor_steps:
                self._terminate(state, "step_limit")
                return
            if self.estimate(self.config.anchor_model, prompt, self.config.anchor_output) > state.remaining:
                self._terminate(state, "budget")
                return
            feature_texts = {"task": state.task.question, **spans(state)}
            if state.candidate is not None:
                feature_texts["candidate"] = state.candidate
            feature_texts = {key:value for key,value in feature_texts.items() if key in packet.feature_spans}
            request = GenerationRequest(self.config.anchor_model, "anchor", prompt,
                                        self.config.anchor_output, state.calls["anchor"], feature_texts, state.seed, packet.feature_spans)
            if not self.fits_context(request):
                self._terminate(state, "context_limit")
                return
            generation = self.backend.generate(request)
            state.calls["anchor"] += 1
            if generation.usage.model_id != request.model_id:
                raise ValueError("Backend reported usage for the wrong model")
            state.ledger.charge(generation.usage)
            try:
                if generation.usage.output_tokens > request.max_new_tokens:
                    raise ValueError("Anchor output exceeds quota")
                protocol_text = generation.text
                if self.config.repair_protocol_json:
                    from .protocol_json import repair_json
                    protocol_text, _ = repair_json(protocol_text)
                record = AnchorRecord.parse(protocol_text)
                known = set(spans(state)) | ({payload[0]} if payload else set())
                if any(ref not in known for ref in record.refs):
                    raise ValueError("Unknown Anchor reference")
                if record.help_request:
                    req = record.help_request
                    sid = f"s{len(state.steps)+1}"
                    target = sid if req.target == "current_step" else req.target
                    if target not in known | {sid} or any(ref not in known | {sid} for ref in req.required_refs):
                        raise ValueError("Unknown assistance target or required reference")
                    record = replace(record, help_request=replace(req, target=target))
            except ValueError as exc:
                last_error = str(exc)
                state.events.append(CallEvent("anchor", prompt, generation.text, False, str(exc), asdict(generation.usage), generation.latency_ms, generation.raw_text))
                continue
            refs = list(dict.fromkeys(record.refs))
            record = replace(record, refs=refs)
            sid = f"s{len(state.steps)+1}"
            state.steps.append(Step(sid, record))
            state.help_request = record.help_request or HelpRequest(
                "Identify potential errors, omissions, or unresolved issues in the current reasoning and provide useful assistance.",
                sid, list(record.refs), True, "anchor_help_request_null_or_missing")
            if record.invalidate_answer:
                state.candidate = None
                state.features.pop("candidate", None)
            if record.answer is not None:
                state.candidate = record.answer
            state.features.update({k: v for k, v in generation.features.items() if k not in {"output", "help_request", "goal", "answer"}})
            if record.answer is not None or state.candidate is None:
                state.features.pop("candidate", None)
            state.features.pop("help_request", None)
            state.features.pop("current_goal", None)
            if record.help_request and "help_request" in generation.features:
                state.features["help_request"] = generation.features["help_request"]
            if record.goal and "goal" in generation.features:
                state.features["current_goal"] = generation.features["goal"]
            if record.answer and "answer" in generation.features:
                state.features["candidate"] = generation.features["answer"]
            if "output" in generation.features:
                state.features[sid] = generation.features["output"]
                if record.goal:
                    state.features["g" + sid[1:]] = generation.features["output"]
            explicit_refs = set(refs)
            if record.help_request:
                explicit_refs.update(record.help_request.required_refs)
                explicit_refs.add(record.help_request.target)
            if payload and payload[0] in explicit_refs:
                state.artifacts[payload[0]] = payload[1]
                if payload_feature:
                    state.features[payload[0]] = payload_feature
            state.events.append(CallEvent("anchor", prompt, generation.text, True, usage=asdict(generation.usage), latency_ms=generation.latency_ms, raw_output=generation.raw_text))
            if len(state.steps) >= self.config.max_anchor_steps:
                self._terminate(state, "step_limit")
            return
        self._terminate(state, "anchor_format")

    def _specialist(self, state: EpisodeState) -> None:
        request = self._specialist_request(state)
        generation = self.backend.generate(request)
        state.calls["specialist"] += 1
        if generation.usage.model_id != request.model_id:
            raise ValueError("Backend reported usage for the wrong model")
        state.ledger.charge(generation.usage)
        session = state.help
        assert session is not None
        session.stage += 1
        error = None
        try:
            if generation.usage.output_tokens > request.max_new_tokens:
                raise ValueError("Specialist output exceeds quota")
            protocol_text = generation.text
            if self.config.repair_protocol_json:
                from .protocol_json import repair_json
                protocol_text, _ = repair_json(protocol_text)
            parsed = SpecialistRecord.parse(protocol_text)
            if self.backend.count_text_tokens(request.model_id, parsed.draft.text) > self.config.draft_limit:
                raise ValueError("Draft payload exceeds quota")
            session.draft = parsed.draft
            session.private_outputs.append(protocol_text)
        except ValueError as exc:
            error = str(exc)
        for key in ("output", "draft"):
            if error is None and key in generation.features:
                name = f"helper_stage_{session.stage}" if key == "output" else "helper_draft"
                state.features[name] = generation.features[key]
        state.events.append(CallEvent("specialist", request.prompt, generation.text, error is None, error, asdict(generation.usage), generation.latency_ms, generation.raw_text))

    def step(self, prefix: EpisodeState, action: Action) -> EpisodeState:
        if action not in self.actions(prefix):
            raise ValueError(f"Illegal action: {action.key}")
        state = deepcopy(prefix)
        state.decisions += 1
        if action.kind == Kind.FINISH:
            self._terminate(state, "finish")
        elif action.kind in (Kind.CONTINUE, Kind.REPAIR):
            self._anchor(state, CONTINUE_INSTRUCTION if action.kind == Kind.CONTINUE else self.repair_instruction(state))
        elif action.kind == Kind.HELP:
            state.help_count += 1
            state.help = HelpSession(f"h{state.help_count}", action)
            self._specialist(state)
        elif action.kind == Kind.EXPAND:
            self._specialist(state)
        elif action.kind in (Kind.SUBMIT, Kind.ABORT):
            session = state.help
            assert session is not None
            payload = (session.help_id, session.draft.text) if action.kind == Kind.SUBMIT and session.draft else None
            feature = state.features.get("helper_draft")
            state.assistance_log.append({"help_id": session.help_id, "request": state.help_request.__dict__ if state.help_request else None,
                                         "action": action.kind.value, "draft": session.draft.text if session.draft else None,
                                         "stages": session.stage, "submitted": payload is not None})
            state.help = None
            state.features = {k: v for k, v in state.features.items() if not k.startswith("helper_")}
            self._anchor(state, SUBMIT_INSTRUCTION if payload else CONTINUE_INSTRUCTION, payload, feature)
        if not state.terminal and state.decisions >= self.config.max_decisions:
            self._terminate(state, "decision_limit")
        return state

    def run_from(self, prefix: EpisodeState, policy: Callable) -> EpisodeState:
        state = deepcopy(prefix)
        for _ in range(max(0, self.config.max_decisions-state.decisions)):
            if state.terminal:
                return state
            actions = self.actions(state)
            if not actions:
                self._terminate(state, "no_feasible_action")
                return state
            state = self.decide_step(state, policy)
        self._terminate(state, "decision_limit")
        return state

    def decide_step(self, prefix: EpisodeState, policy: Callable) -> EpisodeState:
        """One charged policy decision, also used to visit offline prefixes."""
        state = deepcopy(prefix)
        actions = self.actions(state)
        if not actions:
            self._terminate(state, "no_feasible_action")
            return state
        choice = policy(state, actions)
        if isinstance(choice, Decision):
            if choice.resource_cost is None:
                state.ledger.unmeasured_overheads.append("controller")
            else:
                state.ledger.charge_overhead("controller", choice.resource_cost, choice.unit or "")
            action = choice.action
        else:
            action = choice
            state.ledger.unmeasured_overheads.append("policy_selection")
        if action not in self.actions(state):
            self._terminate(state, "selection_budget")
            return state
        state.decision_log.append({"legal_actions": [a.key for a in actions], "selected_action": action.key,
                                   "phase": "specialist" if state.help else "anchor", "remaining": state.remaining,
                                   "help_request": asdict(state.help_request) if state.help_request else None,
                                   "feature_sources": {key: {"source": feature.source, "dimension": len(feature.values)} for key, feature in state.features.items()},
                                   "action_scores": getattr(policy, "last_scores", None),
                                   "policy": type(policy).__name__})
        return self.step(state, action)

    def run(self, task: Task, policy: Callable, seed: int = 0) -> EpisodeState:
        return self.run_from(self.initial(task, seed), policy)
