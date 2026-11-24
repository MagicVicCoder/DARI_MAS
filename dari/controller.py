"""Small action-conditioned dual-value network over existing activations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .context import build_pack
from .runtime import Decision, EpisodeState, Runtime
from .schema import Action, Kind


@dataclass(frozen=True)
class ControllerConfig:
    anchor_dim: int
    specialist_dim: int
    dim: int = 256
    heads: int = 4
    ff_dim: int = 512
    layers: int = 2
    numeric_dim: int = 12
    action_numeric_dim: int = 8
    lambda_cost: float = .1
    schema_version: int = 2


class ValueController(nn.Module):
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        d = config.dim
        self.anchor_proj = nn.Linear(config.anchor_dim, d)
        self.specialist_proj = nn.Linear(config.specialist_dim, d)
        self.numeric_proj = nn.Linear(config.numeric_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.positions = nn.Embedding(512, d)
        self.sources = nn.Embedding(4, d)
        block = nn.TransformerEncoderLayer(d, config.heads, config.ff_dim,
            dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(block, config.layers, enable_nested_tensor=False)
        self.kind_embed = nn.Embedding(len(Kind), 32)
        self.request_proj = nn.Linear(config.anchor_dim, 8)
        self.width_embed = nn.Embedding(3, 8)
        self.quota_embed = nn.Embedding(3, 8)
        self.action_encoder = nn.Sequential(nn.Linear(56 + 2*d + config.action_numeric_dim, d), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(3*d, d), nn.GELU())
        self.reward_head = nn.Linear(d, 1)
        self.cost_head = nn.Linear(d, 1)

    def forward(self, observation: dict[str, Tensor], actions: dict[str, Tensor]) -> dict[str, Tensor]:
        legal = actions["legal"].bool()
        if legal.ndim != 2 or not legal.any(dim=1).all():
            raise ValueError("Each batch item needs at least one legal action")
        # Detached backbone observations: theta trains without gradients into A/S.
        a = self.anchor_proj(observation["anchor"].detach()) + self.sources.weight[0]
        s = self.specialist_proj(observation["specialist"].detach()) + self.sources.weight[1]
        n = self.numeric_proj(observation["numeric"].detach()).unsqueeze(1) + self.sources.weight[2]
        batch = a.shape[0]
        tokens = torch.cat([self.cls.expand(batch, -1, -1), n, a, s], dim=1)
        if tokens.shape[1] > self.positions.num_embeddings:
            raise ValueError("Too many state vectors")
        tokens = tokens + self.positions(torch.arange(tokens.shape[1], device=tokens.device))[None]
        fixed = torch.ones(batch, 2, dtype=torch.bool, device=tokens.device)
        valid = torch.cat([fixed, observation["anchor_valid"], observation["specialist_valid"]], dim=1)
        state = self.fusion(tokens, src_key_padding_mask=~valid)[:, 0]
        target = self.anchor_proj(actions["target"].detach())
        context = (self.anchor_proj(actions["context_anchor"].detach())
                   + self.specialist_proj(actions["context_specialist"].detach()))
        categorical = torch.cat([self.kind_embed(actions["kind"]),
            self.request_proj(actions["request"].detach()), self.width_embed(actions["width"]),
            self.quota_embed(actions["quota"])], dim=-1)
        action = self.action_encoder(torch.cat([categorical, target, context, actions["numeric"].detach()], dim=-1))
        state = state[:, None, :].expand_as(action)
        joint = self.scorer(torch.cat([state, action, state*action], dim=-1))
        reward = torch.sigmoid(self.reward_head(joint).squeeze(-1))
        cost = F.softplus(self.cost_head(joint).squeeze(-1))
        utility = (reward - self.config.lambda_cost*cost).masked_fill(~legal, -torch.inf)
        return {"reward": reward, "cost": cost, "utility": utility}


def encode_state_actions(runtime: Runtime, state: EpisodeState, actions: list[Action],
                         anchor_dim: int, specialist_dim: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Feature extraction uses cached vectors only; no text/model re-encoding."""
    if not actions:
        raise ValueError("Cannot encode an empty action set")

    def vector(key: str, source: str, dim: int) -> list[float]:
        feature = state.features.get(key)
        if feature is None or feature.source != source:
            return [0.]*dim
        if len(feature.values) != dim:
            raise ValueError(f"Wrong hidden dimension for {key}")
        return feature.values

    akeys = (["task"] if "task" in state.features else []) + [s.step_id for s in state.steps[-runtime.config.controller_history:]] + ["current_goal", "help_request"]
    if state.candidate is not None:
        akeys.append("candidate" if "candidate" in state.features else state.steps[-1].step_id)
    skeys = [f"helper_stage_{i}" for i in range(1, runtime.config.max_specialist_stages+1) if f"helper_stage_{i}" in state.features]
    if "helper_draft" in state.features:
        skeys.append("helper_draft")
    anchor = torch.zeros(1, runtime.config.controller_history+4, anchor_dim)
    specialist = torch.zeros(1, runtime.config.max_specialist_stages+1, specialist_dim)
    av, sv = torch.zeros(1, runtime.config.controller_history+4, dtype=torch.bool), torch.zeros(1, runtime.config.max_specialist_stages+1, dtype=torch.bool)
    for i, key in enumerate(akeys):
        anchor[0, i] = torch.tensor(vector(key, "anchor", anchor_dim))
        av[0, i] = key in state.features
    for i, key in enumerate(skeys):
        specialist[0, i] = torch.tensor(vector(key, "specialist", specialist_dim))
        sv[0, i] = True
    cfg = runtime.config
    numeric = [state.remaining/max(state.budget, 1.), state.ledger.total/cfg.reference_cost,
        float(state.help is not None), (state.help.stage/cfg.max_specialist_stages if state.help else 0.),
        len(state.steps)/cfg.max_anchor_steps, state.calls["specialist"]/(2*cfg.max_anchor_steps),
        float(state.candidate is not None), state.calls["anchor"]/cfg.max_anchor_steps,
        state.help_count/cfg.max_anchor_steps, state.budget/cfg.reference_cost,
        float(state.help_request.fallback if state.help_request else False), state.decisions/cfg.max_decisions]
    obs = {"anchor": anchor, "specialist": specialist, "anchor_valid": av, "specialist_valid": sv,
           "numeric": torch.tensor([numeric], dtype=torch.float32)}
    kinds, templates, widths, quotas, targets, contexts_a, contexts_s, numbers = [], [], [], [], [], [], [], []
    legal_set = set(runtime.actions(state))
    for action in actions:
        kinds.append(list(Kind).index(action.kind))
        templates.append(0)
        widths.append({"": 0, "narrow": 1, "wide": 2}[action.width])
        quotas.append({"": 0, "short": 1, "oneshot": 2}[action.quota])
        targets.append(vector(action.target, "anchor", anchor_dim))
        ca, cs = [], []
        input_length, output_cap, estimated_cost, count = 0, 0, 0., 0
        req = runtime.request_for_action(state, action)
        if req is not None:
            input_length, output_cap = runtime.backend.count_tokens(req.model_id, req.prompt), req.max_new_tokens
            estimated_cost = runtime.estimate(req.model_id, req.prompt, req.max_new_tokens)
        if action.kind == Kind.HELP:
            pack = build_pack(state, action.target, action.width)
            count = len(pack.span_ids)
            for key in pack.span_ids:
                feature = state.features.get(key)
                if feature and feature.source == "anchor":
                    ca.append(vector(key, "anchor", anchor_dim))
                elif feature and feature.source == "specialist":
                    cs.append(vector(key, "specialist", specialist_dim))
        contexts_a.append(torch.tensor(ca).mean(0).tolist() if ca else [0.]*anchor_dim)
        contexts_s.append(torch.tensor(cs).mean(0).tolist() if cs else [0.]*specialist_dim)
        numbers.append([input_length/cfg.specialist_input_limit, output_cap/cfg.oneshot_output,
                        estimated_cost/cfg.reference_cost, count/16., float(action.kind == Kind.SUBMIT),
                        float(action.kind == Kind.ABORT), float(action.kind == Kind.FINISH),
                        state.remaining/cfg.reference_cost])
    request_key = state.help_request.target if state.help_request and state.help_request.fallback else "help_request"
    encoded = {
        "kind": torch.tensor([kinds]), "template": torch.tensor([templates]),
        "request": torch.tensor([[vector(request_key, "anchor", anchor_dim) for _ in actions]], dtype=torch.float32),
        "width": torch.tensor([widths]), "quota": torch.tensor([quotas]),
        "target": torch.tensor([targets], dtype=torch.float32),
        "context_anchor": torch.tensor([contexts_a], dtype=torch.float32),
        "context_specialist": torch.tensor([contexts_s], dtype=torch.float32),
        "numeric": torch.tensor([numbers], dtype=torch.float32),
        "legal": torch.tensor([[a in legal_set for a in actions]], dtype=torch.bool),
    }
    return obs, encoded


class NeuralPolicy:
    """Optional deployed scorer. Supply an explicit resource-overhead calibrator.

    Without one, the ledger marks controller cost unmeasured. CPU elapsed time
    must not be silently added to a GPU-ms or dollar budget.
    """
    def __init__(self, runtime: Runtime, model: ValueController,
                 overhead: Callable[[], tuple[float, str]] | None = None):
        self.runtime, self.model, self.overhead = runtime, model, overhead

    def __call__(self, state: EpisodeState, actions: list[Action]) -> Decision:
        if not self.runtime.backend.synthetic:
            if not state.features.get("task") or not state.steps or state.steps[-1].step_id not in state.features:
                raise ValueError("Neural Controller requires actual model hidden states")
            if state.help_request and not state.help_request.fallback and "help_request" not in state.features:
                raise ValueError("Dynamic request hidden span is unavailable; cannot substitute a fabricated feature")
        cfg = self.model.config
        obs, encoded = encode_state_actions(self.runtime, state, actions, cfg.anchor_dim, cfg.specialist_dim)
        device = next(self.model.parameters()).device
        obs = {k: v.to(device) for k, v in obs.items()}
        encoded = {k: v.to(device) for k, v in encoded.items()}
        self.model.eval()
        with torch.inference_mode():
            values = self.model(obs, encoded)
            self.last_scores = {key: value.cpu().tolist() for key, value in values.items()}
            index = values["utility"].argmax(-1).item()
        if self.overhead is None:
            return Decision(actions[index])
        cost, unit = self.overhead()
        return Decision(actions[index], cost, unit)
