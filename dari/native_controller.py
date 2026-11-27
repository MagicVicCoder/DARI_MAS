"""Schema 4 observation adapter for the existing action-conditioned network."""
import torch

from .schema import Kind

TARGETS = ("", "current_frontier", "previous_frontier", "current_candidate", "native_open_issue")
MODES = ("auto", "advance", "verify", "explore")


def score_actions(runtime, state, model, actions=None):
    cfg = model.config
    if cfg.schema_version != 4:
        raise ValueError("Native scoring requires Controller schema 4")
    legal = runtime.actions(state)
    actions = legal if actions is None else actions
    if not actions or not any(action in legal for action in actions):
        raise ValueError("No legal actions to score")
    anchor = state.anchor_hidden or [0.] * cfg.anchor_dim
    output = state.session.features if state.session else {}
    specialist = [output[key].values if key in output else [0.] * cfg.specialist_dim
                  for key in ("output", "draft")]
    device = next(model.parameters()).device
    tensor = lambda value, **kwargs: torch.tensor(value, device=device, **kwargs)
    observation = {
        "anchor": tensor([[anchor]], dtype=torch.float32),
        "anchor_valid": tensor([[bool(state.anchor_hidden)]]),
        "specialist": tensor([specialist], dtype=torch.float32),
        "specialist_valid": tensor([[key in output for key in ("output", "draft")]]),
        "numeric": tensor([[len(state.generated_ids) / runtime.config.anchor_budget,
                            state.specialist_tokens / max(1, runtime.config.specialist_budget),
                            len(state.injections), float(state.candidate is not None),
                            *([0.] * 8)]], dtype=torch.float32),
    }
    objects = runtime.objects(state)
    numeric = []
    targets = []
    for action in actions:
        obj = objects.get(action.target or "current_frontier") if action.kind == "help" else None
        begin, end = obj.source_token_span if obj else (0, 0)
        targets.append(obj.hidden if obj and obj.hidden else [0.] * cfg.anchor_dim)
        numeric.append([TARGETS.index(action.target), MODES.index(action.mode),
                        (end - begin) / runtime.config.anchor_budget,
                        (state.position - end) / runtime.config.anchor_budget if obj else 0.,
                        float(action.kind == "help"), 0., 0., 0.])
    encoded = {
        "legal": tensor([[action in legal for action in actions]]),
        "kind": tensor([[next(i for i, kind in enumerate(Kind) if kind.value == action.kind)
                         for action in actions]]),
        "width": tensor([[0] * len(actions)]),
        "quota": tensor([[0] * len(actions)]),
        "numeric": tensor([numeric], dtype=torch.float32),
    }
    encoded["target"] = tensor([targets], dtype=torch.float32)
    for name, dimension in (("request", cfg.anchor_dim),
                            ("context_anchor", cfg.anchor_dim), ("context_specialist", cfg.specialist_dim)):
        encoded[name] = torch.zeros((1, len(actions), dimension), device=device)
    return model(observation, encoded)
