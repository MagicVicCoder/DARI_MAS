"""CPU controller trainer for JSONL branch targets; no model downloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch

from .collection import read_targets
from .controller import ControllerConfig, ValueController
from .training import controller_loss


def _tensors(rows, field):
    first = getattr(rows[0], field)
    result = {}
    for name in first:
        dtype = torch.bool if name.endswith("valid") or name == "legal" else torch.long if name in {"kind", "template", "width", "quota"} else torch.float32
        result[name] = torch.cat([torch.tensor(getattr(row, field)[name], dtype=dtype) for row in rows], dim=0)
    return result


def train_from_targets(data: str | Path, output: str | Path, steps: int = 20,
                       dim: int = 256, seed: int = 7, resume: str | Path | None = None,
                       batch_size: int = 64, lr: float = .003, device: str = "cpu",
                       lambda_cost: float = .1, allow_specialist_update: bool = False) -> dict:
    if steps <= 0 or dim <= 0 or dim % 4 or batch_size <= 0 or lr <= 0:
        raise ValueError("Positive steps and dimension divisible by 4 required")
    rows = read_targets(data)
    if not rows:
        raise ValueError("Empty target file")
    signature = {(r.cost_unit, r.reference_cost, r.deployment_id, r.snapshot_id, r.synthetic) for r in rows}
    if len(signature) != 1:
        raise ValueError("Mixed deployments, cost normalization, units or policy snapshots: recollect/filter data")
    if any(not r.cost_complete for r in rows):
        raise ValueError("Incomplete cost labels cannot train a cost-aware controller")
    torch.manual_seed(seed)
    obs = _tensors(rows[:1], "observation")
    config = ControllerConfig(obs["anchor"].shape[-1], obs["specialist"].shape[-1], dim=dim, ff_dim=2*dim)
    config = ControllerConfig(config.anchor_dim, config.specialist_dim, dim=dim, ff_dim=2*dim, lambda_cost=lambda_cost)
    model = ValueController(config).to(device)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"] != asdict(config):
            raise ValueError("Warm-start controller architecture/cost weight mismatch")
        old = saved["data_metadata"]
        compatible = old["deployment_id"] == rows[0].deployment_id
        if allow_specialist_update and rows[0].base_deployment_id:
            compatible = compatible or old.get("base_deployment_id") == rows[0].base_deployment_id
        if not compatible or any(old[k] != getattr(rows[0], k) for k in ("synthetic", "cost_unit", "reference_cost")):
            raise ValueError("Warm-start controller deployment/provenance mismatch")
        model.load_state_dict(saved["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)

    def batch(selected):
        obs = {k: v.to(device) for k, v in _tensors(selected, "observation").items()}
        actions = {k: v.to(device) for k, v in _tensors(selected, "action_features").items()}
        reward = torch.tensor([[r.reward] for r in selected], device=device)
        cost = torch.tensor([[r.cost] for r in selected], device=device)
        return obs, actions, reward, cost
    losses = []
    for _ in range(steps):
        selected = [rows[i] for i in torch.randint(len(rows), (min(batch_size, len(rows)),)).tolist()]
        obs, actions, reward, cost = batch(selected)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(obs, actions)
        loss = controller_loss(predictions, reward, cost, actions["legal"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start+batch_size]
            obs, actions, reward, cost = batch(selected)
            total_loss += float(controller_loss(model(obs, actions), reward, cost, actions["legal"]))*len(selected)
        final_loss = total_loss/len(rows)
    metadata = {"snapshot_id": rows[0].snapshot_id, "deployment_id": rows[0].deployment_id,
                "base_deployment_id": rows[0].base_deployment_id,
                "cost_unit": rows[0].cost_unit, "reference_cost": rows[0].reference_cost,
                "synthetic": rows[0].synthetic, "examples": len(rows)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({"model_state": model.state_dict(), "config": asdict(config), "data_metadata": metadata}, temporary)
    temporary.replace(output)
    return {"initial_loss": losses[0], "final_loss": final_loss, "steps": steps,
            "parameters": sum(p.numel() for p in model.parameters()), "checkpoint": str(output),
            "initialized_from": str(resume) if resume else None, **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-cost", type=float, default=.1)
    parser.add_argument("--allow-specialist-update", action="store_true", help="Warm-start across phi updates with identical frozen backbones/costs")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    print(json.dumps(train_from_targets(args.data, args.output, args.steps, args.dim,
        resume=args.resume, batch_size=args.batch_size, lr=args.lr, device=args.device,
        lambda_cost=args.lambda_cost, allow_specialist_update=args.allow_specialist_update), indent=2))


if __name__ == "__main__":
    main()
