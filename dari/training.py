"""Differentiable theta/phi objectives; no gradient through receiver rollouts."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def controller_loss(prediction: dict[str, Tensor], reward_target: Tensor,
                    cost_target: Tensor, valid: Tensor, eta_cost: float = 1.) -> Tensor:
    if not valid.any():
        raise ValueError("No labeled action")
    if not torch.isfinite(reward_target[valid]).all() or not torch.isfinite(cost_target[valid]).all():
        raise ValueError("Targets must be finite")
    if ((reward_target[valid] < 0) | (reward_target[valid] > 1)).any() or (cost_target[valid] < 0).any():
        raise ValueError("Reward must be in [0,1], normalized cost nonnegative")
    return (F.smooth_l1_loss(prediction["reward"][valid], reward_target[valid].detach())
            + eta_cost * F.smooth_l1_loss(prediction["cost"][valid], cost_target[valid].detach()))


def sequence_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """Causal log likelihood; prompt/padding labels must be -100."""
    targets = labels[:, 1:]
    valid = targets != -100
    if not valid.any(dim=1).all():
        raise ValueError("Each example needs a response token")
    # Avoid a float32 log-softmax over the entire long prompt. Only response
    # positions contribute to the objective; preserve one sum per sequence.
    selected = logits[:, :-1][valid].float()
    logps = F.log_softmax(selected, dim=-1).gather(-1, targets[valid].unsqueeze(-1)).squeeze(-1)
    owners = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(targets)[valid]
    return logps.new_zeros(labels.shape[0]).scatter_add(0, owners, logps)


def sft_loss(logits: Tensor, labels: Tensor) -> Tensor:
    valid_count = (labels[:, 1:] != -100).sum()
    if valid_count == 0:
        raise ValueError("No response tokens")
    return -sequence_log_probs(logits, labels).sum() / valid_count


def dpo_loss(chosen: Tensor, rejected: Tensor, ref_chosen: Tensor, ref_rejected: Tensor,
             beta: float = .1) -> Tensor:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be positive")
    margin = (chosen - rejected) - (ref_chosen.detach() - ref_rejected.detach())
    return -F.logsigmoid(beta * margin).mean()


class LoRALinear(nn.Module):
    """Minimal unfused LoRA for ordinary attention Linear projections.

    It deliberately does not claim PEFT checkpoint compatibility or support
    quantized/fused projection classes. Those need a backend-specific adapter.
    """
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.):
        super().__init__()
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid LoRA rank/alpha")
        self.base, self.rank, self.alpha = base, rank, alpha
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: Tensor) -> Tensor:
        update = F.linear(F.linear(value, self.lora_A), self.lora_B)
        return self.base(value) + (self.alpha/self.rank)*update


def inject_lora(model: nn.Module, targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
                rank: int = 16, alpha: float = 32.) -> list[str]:
    selected = [(name, layer) for name, layer in model.named_modules()
                if name and name.rsplit(".", 1)[-1] in targets and isinstance(layer, nn.Linear)]
    if not selected:
        raise ValueError("No ordinary Linear attention projections matched; supply a model-specific adapter")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, layer in selected:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(layer, rank, alpha))
    return [name for name, _ in selected]
