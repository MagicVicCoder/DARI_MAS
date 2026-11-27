"""Audit executed branch targets before offline training."""
from collections import Counter
import json
import math
import statistics

import torch

from .collection import read_targets
from .phase3a import ROOT, state_id, write_json


def audit(split):
    counts = Counter()
    completed = set()
    for path in sorted((ROOT / "collection" / split).rglob("targets.jsonl")):
        targets = read_targets(path)
        prefix = torch.load(path.parent / "prefix.pt", weights_only=False)
        prefix_hash = state_id(prefix)
        if not targets or prefix_hash in completed:
            raise ValueError(f"Empty or duplicate group {path}")
        completed.add(prefix_hash)
        outcomes = [json.loads(file.read_text()) for file in path.parent.glob("*.json")]
        if {target.action.key for target in targets} != set(targets[0].provenance["sampled_actions"]):
            raise ValueError(f"Action coverage mismatch {path}")
        for target in targets:
            action = target.action
            matches = [row for row in outcomes if row["action"]["kind"] == action.kind.value
                       and row["action"]["quota"] == action.quota and row["action"]["width"] == action.width]
            assert target.prefix_id == prefix_hash
            assert len(matches) == target.repeats and target.action_features["legal"] == [[True]]
            assert all(row["future_ms"] >= 0 and row["shared_ms"] == prefix.ledger.total for row in matches)
            assert math.isclose(target.cost, statistics.mean(row["future_ms"] for row in matches) / target.reference_cost)
            assert math.isclose(target.reward, statistics.mean(row["reward"]["reward"] for row in matches))
            counts["rollouts"] += len(matches)
            counts[action.kind.value] += 1
        counts[targets[0].provenance["phase"] + "_states"] += 1
    return dict(counts)


if __name__ == "__main__":
    result = {split: audit(split) for split in ("train", "dev")}
    if not result["train"].get("anchor_states") or not result["train"].get("specialist_states"):
        raise ValueError("The Pilot must produce paired Anchor and Specialist targets")
    write_json(ROOT / "integrity_audit.json", result)
    print(result)
