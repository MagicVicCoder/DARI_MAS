"""Export paired historical evidence without combining scoring protocols."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json

import torch

from .phase3b import ROOT
from .pilot import read_rows


def main():
    groups = [
        ("phase2a", "phase2a/e1", "phase2a"),
        ("refinement_r2", "phase2a_refinement/r2/e1", "phase2a"),
        ("refinement_r3", "phase2a_refinement/r3_branches/e1", "phase2a_refinement/r3"),
        ("phase2b", "phase2b/t1/e1", "phase2b/t1"),
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    cases, summaries, examples = [], {}, []
    for phase, results, samples in groups:
        directory = ROOT.parent / results
        questions = {row["task_id"]: row for row in read_rows(ROOT.parent / samples / "samples.jsonl")}
        key = lambda row: (row["prefix_id"], row["method"], row.get("repeat", 0))
        traces = {key(row): row for row in read_rows(directory / "traces.jsonl")}
        pairs = defaultdict(dict)
        for row in read_rows(directory / "results.jsonl"):
            if row.get("status") == "completed":
                pairs[row["prefix_id"], row.get("repeat", 0)][row["method"]] = row
        stats = defaultdict(Counter)
        for (prefix_id, repeat), branches in sorted(pairs.items()):
            for method, helped in branches.items():
                if not method.startswith("help") or "continue" not in branches:
                    continue
                controls = [branches.get(name) for name in ("continue", "self_repair", "self_repair_extended")]
                reward = helped["reward"]
                unique = reward == 1 and all(row is not None and row["reward"] < 1 for row in controls)
                delta = reward - controls[0]["reward"]
                counts = stats[helped["dataset"] + "/" + method]
                counts["paired"] += 1
                counts["positive" if delta > 0 else "negative" if delta < 0 else "tie"] += 1
                counts["unique_success"] += unique
                if controls[2] is not None:
                    counts["beats_extended"] += reward > controls[2]["reward"]
                task = questions[helped["task_id"]]
                case = {"phase": phase, "prefix_id": prefix_id, "repeat": repeat, "method": method,
                        "unique_success": unique, "delta_reward": delta, "task": task,
                        "branches": {name: {"result": row, "trace": traces.get(key(row))}
                                     for name, row in branches.items()}}
                snapshot = directory / "prefixes" / (prefix_id + ".pt")
                if snapshot.is_file():
                    prefix = torch.load(snapshot, weights_only=False, map_location="cpu")
                    case["shared_state"] = {"steps": [asdict(step) for step in prefix.steps],
                                            "candidate": prefix.candidate,
                                            "request": asdict(prefix.help_request) if prefix.help_request else None}
                cases.append(case)
                if unique:
                    examples += [f"## {phase} · {task['task_id']} · {method}", "",
                                 task["question"], "", f"参考答案 `{task['gold']}`", "",
                                 "完整输入和原始证据见 historical_cases.jsonl。", ""]
                    shared = case.get("shared_state", {})
                    examples += ["共享前缀与动态请求", "", "```json",
                                 json.dumps(shared, ensure_ascii=False, indent=2), "```", ""]
                    for name in ("continue", "self_repair", "self_repair_extended", method):
                        branch = case["branches"][name]
                        examples += [f"### {name}", "",
                                     f"终局奖励 {branch['result']['reward']}，"
                                     f"最终答案 `{branch['result'].get('answer')}`。", ""]
                        for event in (branch["trace"] or {}).get("events", []):
                            examples += [event["role"], "", "```text",
                                         event.get("output", ""), "```", ""]
        summaries[phase] = stats
    with (ROOT / "historical_cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    (ROOT / "historical_summary.json").write_text(json.dumps(summaries, indent=2))
    report = ROOT / "report.md"
    marker = "\n## 真实配对实验\n"
    existing = report.read_text() if report.exists() else ""
    comparison = marker + existing.split(marker, 1)[1] if marker in existing else ""
    report.write_text("\n".join([
        "# DARI-MAS Phase 3B", "", "阶段状态：历史配对导出，真实机制实验进行中。", "",
        f"已导出 {len(cases)} 个历史 Help 与同前缀内部对照组。各时期分别按原始评分统计。", "",
        *examples]) + "\n" + comparison)


if __name__ == "__main__":
    main()
