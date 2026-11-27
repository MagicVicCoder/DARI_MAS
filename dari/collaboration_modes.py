"""Prompt and information controls for the Phase 3B interventions."""

ROLE_INSTRUCTIONS = {
    "check": "Check the current local reasoning for errors, omissions and bad assumptions. Derive a concrete correction where possible.",
    "recompute": "Independently derive the current local goal from the original problem and evidence. Give the calculation or evidence and conclusion.",
    "solver": "Independently solve the entire original task from the supplied question and sources.",
    "autonomous": "Complete the proposed request if sound. If it is already solved, rests on a false assumption or misses a clear bottleneck, perform the closest useful local task and explain the change in the draft.",
}


def configure_specialist(config, state, body, instruction, tracked, requested):
    role = config.specialist_role
    if role in {"check", "recompute", "solver"}:
        body["dynamic_request"] = ROLE_INSTRUCTIONS[role]
    if role != "directed":
        instruction = ROLE_INSTRUCTIONS[role] + "\n" + instruction
    independent = config.specialist_context == "independent" or (
        config.specialist_context == "compare" and state.help.stage == 0)
    if independent or role == "solver":
        body["anchor_reasoning_context"] = {}
        body["relevant_source_evidence"] = dict(state.task.sources)
        body["selected_context_ids"] = list(state.task.sources)
        tracked = {("original_question_and_constraints",): "task",
                   **{("relevant_source_evidence", key): key for key in state.task.sources}}
        requested = dict(state.task.sources)
    if config.specialist_context == "compare" and state.help.stage:
        body["stage_instruction"] = (
            "Compare your previous independent derivation with the newly supplied Anchor reasoning. "
            "Resolve differences using the original evidence and draft the useful result.")
    return instruction, tracked, requested
