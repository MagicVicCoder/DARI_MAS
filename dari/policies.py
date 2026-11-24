"""Explicit debug policies; their forced actions are not learned decisions."""
from .runtime import Decision
from .schema import Kind


class DebugPolicy:
    def __init__(self, mode="anchor_only", help_sessions=1, minimum_steps=2):
        if mode not in {"anchor_only", "self_repair", "help_submit", "help_expand_submit", "help_abort", "bootstrap"}:
            raise ValueError("Unknown debug policy")
        self.mode, self.help_sessions, self.minimum_steps = mode, help_sessions, minimum_steps

    def __call__(self, state, legal):
        if state.help:
            desired = Kind.ABORT if self.mode == "help_abort" else Kind.SUBMIT
            if self.mode == "help_expand_submit" and state.help.stage < 3:
                desired = Kind.EXPAND
            order = (desired, Kind.SUBMIT, Kind.ABORT)
        elif self.mode.startswith("help") or self.mode == "bootstrap":
            order = (Kind.HELP, Kind.CONTINUE, Kind.FINISH) if state.help_count < self.help_sessions else (Kind.FINISH, Kind.CONTINUE)
        elif self.mode == "self_repair" and len(state.steps) < self.minimum_steps:
            order = (Kind.REPAIR, Kind.CONTINUE, Kind.FINISH)
        else:
            order = (Kind.CONTINUE, Kind.FINISH) if len(state.steps) < self.minimum_steps else (Kind.FINISH, Kind.CONTINUE)
        selected = next((action for kind in order for action in legal if action.kind == kind), legal[0])
        return Decision(selected)


class AnchorOnlyPolicy(DebugPolicy):
    def __init__(self, **kwargs):
        super().__init__("anchor_only", **kwargs)


class ForceSelfRepairPolicy(DebugPolicy):
    def __init__(self, **kwargs):
        super().__init__("self_repair", **kwargs)


class ForceHelpSubmitPolicy(DebugPolicy):
    def __init__(self, **kwargs):
        super().__init__("help_submit", **kwargs)


class ForceHelpExpandSubmitPolicy(DebugPolicy):
    def __init__(self, **kwargs):
        super().__init__("help_expand_submit", **kwargs)


class ForceHelpAbortPolicy(DebugPolicy):
    def __init__(self, **kwargs):
        super().__init__("help_abort", **kwargs)


class BootstrapPolicy(DebugPolicy):
    def __init__(self, **kwargs):
        super().__init__("bootstrap", **kwargs)
