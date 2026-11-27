"""Summarize paired terminal rewards and resources."""
from collections import Counter, defaultdict
import json
from statistics import mean

from .phase3b import CONDITIONS, ROOT
from .phase3a import write_json


def records(split):
    groups = defaultdict(dict)
    for path in (ROOT / split).glob("*/*.json"):
        if path.is_file():
            row = json.loads(path.read_text())
            groups[row["prefix_id"]][row["condition"]] = row
    return groups


def reward(row):
    return row["score"]["reward"]


def choose():
    groups = records("screen")
    screen_conditions = set(CONDITIONS) - {"autonomous_independent_critical"}
    complete = [group for group in groups.values() if screen_conditions <= group.keys()]
    if not complete:
        raise ValueError("No complete screening groups")
    candidates = []
    for mode in ("autonomous", "independent", "independent_compare", "autonomous_independent"):
        deltas = [reward(group[mode])-reward(group["directed"]) for group in complete]
        extra = mean(g[mode]["future_ms"]-g["directed"]["future_ms"] for g in complete)
        candidates.append((sum(delta > 0 for delta in deltas) - 2*sum(delta < 0 for delta in deltas),
                           -extra, mode))
    return [max(candidates)[2]]


def main():
    summary = {}
    for split in ("screen", "integration_history", "fresh_validation", "matched_sampling", "combined_validation"):
        groups = records(split)
        stats = {}
        for condition in sorted({condition for group in groups.values() for condition in group}):
            pairs = [group for group in groups.values() if condition in group]
            rows = [group[condition] for group in pairs]
            baseline = "integration_abort" if condition.startswith("integration_") else "continue"
            deltas = [reward(group[condition])-reward(group[baseline]) for group in pairs if baseline in group]
            controls = [group for group in pairs if {"continue", "repair", "extended"} <= group.keys()]
            internal = [max(reward(g[key]) for key in ("continue", "repair", "extended")) for g in controls]
            stats[condition] = {
                "n": len(rows), "reward": mean(map(reward, rows)), "mean_ms": mean(r["future_ms"] for r in rows),
                "baseline": baseline,
                "win_tie_loss": dict(Counter("win" if d > 0 else "loss" if d < 0 else "tie" for d in deltas)),
                "internal_oracle": sum(internal),
                "with_condition_oracle": sum(max(value, reward(g[condition])) for value, g in zip(internal, controls)),
                "format_failures": sum(r["metrics"]["format_failures"] for r in rows),
                "tokens": {key: sum(usage[key] for row in rows for usage in row["metrics"]["tokens_by_model"].values())
                           for key in ("input", "output", "calls")},
            }
        summary[split] = stats
    write_json(ROOT / "results_summary.json", summary)
    path = ROOT / "report.md"
    marker = "\n## 真实配对实验\n"
    historical = path.read_text().split(marker, 1)[0]
    lines = [marker, "耗时为共享服务器上的实际串行执行时间。独立上下文仍保留动态任务原文。", ""]
    for split, stats in summary.items():
        lines += [f"### {split}", "", "| 条件 | 样本 | 平均奖励 | 平均毫秒 | 格式失败 |",
                  "|---|---:|---:|---:|---:|"]
        for condition, row in stats.items():
            lines.append(f"| {condition} | {row['n']} | {row['reward']:.3f} | "
                         f"{row['mean_ms']:.1f} | {row['format_failures']} |")
        lines.append("")
    path.write_text(historical + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
