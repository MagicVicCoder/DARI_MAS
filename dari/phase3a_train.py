"""Fit and evaluate the existing Controller on grouped Phase 3A targets."""
from collections import Counter, defaultdict
from dataclasses import asdict
import random
import statistics

import torch
from torch.nn import functional as F

from .collection import read_targets
from .controller import ControllerConfig, ValueController
from .phase3a import ROOT, write_json
from .train_controller import _tensors


def load(split):
    return [row for path in sorted((ROOT / "collection" / split).rglob("targets.jsonl"))
            for row in read_targets(path)]


def fit(train, dev, name):
    torch.manual_seed(20261127)
    complete = all(row.cost_complete for row in train + dev)
    numeric = name == "numeric"
    first = train[0]
    config = ControllerConfig(len(first.observation["anchor"][0][0]),
                              len(first.observation["specialist"][0][0]),
                              lambda_cost=.1 if complete else 0.)
    model = ValueController(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def pack(rows):
        obs = _tensors(rows, "observation")
        actions = _tensors(rows, "action_features")
        if numeric:
            for key in ("anchor", "specialist", "anchor_valid", "specialist_valid"):
                obs[key].zero_()
            for key in ("request", "target", "context_anchor", "context_specialist"):
                actions[key].zero_()
        return obs, actions, torch.tensor([[row.reward, row.cost] for row in rows])

    train_obs, train_actions, train_labels = pack(train)
    dev_obs, dev_actions, dev_labels = pack(dev)

    def predict(obs, actions, labels):
        values = model(obs, actions)
        pred = torch.cat((values["reward"], values["cost"]), dim=1)
        loss = F.smooth_l1_loss(pred, labels, reduction="none").mean(0)
        return pred, loss

    def objective(loss):
        return loss.sum() if complete else loss[0]

    output = ROOT / "training" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for step in range(151):
        record = {"step": step, "learning_rate": 3e-4}
        if step:
            model.train()
            indices = torch.randint(len(train), (min(32, len(train)),))
            _, loss = predict({key: tensor[indices] for key, tensor in train_obs.items()},
                              {key: tensor[indices] for key, tensor in train_actions.items()},
                              train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            total = objective(loss)
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.step()
            update = sum(float((parameter.detach()-old).square().sum())
                         for parameter, old in zip(model.parameters(), before)) ** .5
            record.update(train_loss=loss.detach().tolist(), train_total=float(total.detach()),
                          gradient_norm=float(grad), update_norm=update)
        if step % 5 == 0:
            model.eval()
            with torch.no_grad():
                _, val = predict(dev_obs, dev_actions, dev_labels)
            record.update(dev_loss=val.tolist(), dev_total=float(objective(val)))
            if record["dev_total"] < best:
                best, best_step = record["dev_total"], step
                torch.save({"config": asdict(config), "model_state": model.state_dict(),
                            "data_metadata": {"deployment_id": first.deployment_id,
                                              "cost_complete": complete, "numeric_only": numeric}},
                           output.with_suffix(".pt"))
            if step - best_step >= 35:
                history.append(record)
                break
        history.append(record)
    model.load_state_dict(torch.load(output.with_suffix(".pt"), weights_only=True)["model_state"])
    model.eval()
    with torch.no_grad():
        pred, _ = predict(dev_obs, dev_actions, dev_labels)
    write_json(output.with_suffix(".json"), {"history": history, "best_step": best_step,
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "cost_complete": complete})
    return pred.tolist(), config.lambda_cost


def evaluate(rows, predictions, weight, name):
    groups = defaultdict(list)
    bins = defaultdict(list)
    action_errors = defaultdict(list)
    pairs = []
    for row, (reward, cost) in zip(rows, predictions):
        groups[row.prefix_id].append((row, reward - weight * cost))
        bins[min(4, int(reward * 5))].append((reward, row.reward))
        action_errors[row.action.kind.value].append(abs(cost - row.cost))
        pairs.append({"prefix_id": row.prefix_id, "action": row.action.key,
                      "reward": row.reward, "predicted_reward": reward, "cost": row.cost,
                      "predicted_cost": cost})
    rng = random.Random(20261127)
    decisions = defaultdict(list)
    for group in groups.values():
        phase = group[0][0].provenance["phase"]
        utility = lambda item: item[0].reward - weight * item[0].cost
        oracle = max(map(utility, group))
        baseline_order = ("submit", "abort", "expand") if phase == "specialist" else ("finish", "continue", "self_repair", "help")
        fixed = min(group, key=lambda item: baseline_order.index(item[0].action.kind.value))
        restricted = [item for item in group if item[0].action.kind.value != "help" and
                      (phase != "specialist" or item[0].action.kind.value == "submit")]
        for policy, selected in (("random", rng.choice(group)), ("fixed", fixed),
                                 (name, max(group, key=lambda item: item[1])),
                                 ("oracle", max(group, key=utility)),
                                 ("restricted_oracle", max(restricted or group, key=utility)),
                                 ("reward_oracle", max(group, key=lambda item: item[0].reward))):
            decisions[phase + "/" + policy].append((selected[0].action.kind.value,
                selected[0].reward, selected[0].cost, utility(selected), oracle - utility(selected)))
    result = {"reward_mae": statistics.mean(abs(p["reward"]-p["predicted_reward"]) for p in pairs),
              "reward_mse": statistics.mean((p["reward"]-p["predicted_reward"])**2 for p in pairs),
              "reward_calibration": {key: [len(values), *map(statistics.mean, zip(*values))] for key, values in bins.items()},
              "cost_mae_by_action": {key: statistics.mean(values) for key, values in action_errors.items()} if weight else None,
              "cost_relative_mae_nonzero": statistics.mean(abs(p["cost"]-p["predicted_cost"])/p["cost"]
                    for p in pairs if p["cost"] > 0) if weight else None,
              "decisions": {key: {"n": len(values), "action_counts": dict(Counter(v[0] for v in values)),
                   **dict(zip(("reward", "cost", "utility", "regret"), map(statistics.mean, zip(*(v[1:] for v in values))))),
                   "within_utility_tie": sum(v[-1] <= 1e-8 for v in values)}
                            for key, values in decisions.items()},
              "predictions": pairs}
    write_json(ROOT / "training" / f"{name}_offline.json", result)


def main():
    torch.set_num_threads(2)
    train, dev = load("train"), load("dev")
    if not train or not dev:
        raise ValueError("Training and development targets are required")
    task_ids = lambda rows: {row.provenance["task_id"] for row in rows}
    signature = {(r.deployment_id, r.snapshot_id, r.reference_cost, r.cost_unit, r.cost_complete) for r in train + dev}
    if task_ids(train) & task_ids(dev) or len(signature) != 1:
        raise ValueError("Task leakage or incompatible target provenance")
    for name in ("full", "numeric"):
        predictions, weight = fit(train, dev, name)
        evaluate(dev, predictions, weight, name)


if __name__ == "__main__":
    main()
