"""Summarize only completed Phase 3A artifacts."""
from collections import Counter, defaultdict
import itertools
import json
import statistics

from .phase3a import ROOT
from .phase3a_train import load


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibration = json.loads((ROOT / "c0/calibration.json").read_text())
    lines = ["# DARI-MAS Phase 3A", "", "## C0 资源成本", "",
             f"独立验证平均相对误差 {calibration['mean_validation_relative_error']:.2%}。"
             f"成本确认状态为 {calibration['calibration_confirmed']}。"]
    lines.append("成本单位为串行执行毫秒，覆盖模型调用和同步的策略执行耗时，模型加载在计时外。"
                 "Anchor 与 Specialist 分别拟合，实际标签使用同步墙钟时间并扣除共享历史。"
                 "GPU 共享负载可能改变测量值，独占 GPU 时间和跨设备资源用量需要另行测量。"
                 "每 10 秒归一化为 1。")
    for model, entries in calibration["validation"].items():
        lines.append(f"- {model} 验证 MAE {statistics.mean(row['absolute_error_ms'] for row in entries):.2f} ms。")
    lines += ["", "## C1–C2 实测数据", "",
              "A/S 均使用冻结的原始 Qwen3-8B。评价和语法修复沿用 Phase 2B。"
              "分支采用 Narrow Short Help，两阶段上限，固定续推优先 Finish，否则 Continue，"
              "Specialist 优先提交合法 Draft，否则 Abort。未覆盖 Wide 和第三阶段。"
              "单次 rollout 标签是 Monte Carlo 观察值。"]
    for split in ("train", "dev"):
        rows = load(split)
        groups = defaultdict(dict)
        distributions = defaultdict(list)
        for row in rows:
            groups[row.prefix_id][row.action.kind.value] = row.reward
            distributions[row.provenance["phase"], row.action.kind.value].append(row)
        counts = Counter(group[0].provenance["phase"] for group in
                         {row.prefix_id: [row] for row in rows}.values())
        lines += ["", f"- {split} 实际覆盖 {len({r.provenance['task_id'] for r in rows})} 题，"
                        f"{dict(counts)}，{sum(r.repeats for r in rows)} 次 rollout。"]
        for (phase, action), items in sorted(distributions.items()):
            lines.append(f"- {phase}/{action} 共 {len(items)} 项，奖励均值 "
                         f"{statistics.mean(r.reward for r in items):.3f}，未来耗时均值 "
                         f"{statistics.mean(r.cost*r.reference_cost for r in items):.1f} ms。")
        comparisons = defaultdict(Counter)
        for group in groups.values():
            for left, right in itertools.combinations(sorted(group), 2):
                delta = group[left] - group[right]
                comparisons[left + "/" + right]["win" if delta > 0 else "loss" if delta < 0 else "tie"] += 1
        for pair, tally in sorted(comparisons.items()):
            lines.append(f"- {pair} 同前缀胜/平/负 {tally['win']}/{tally['tie']}/{tally['loss']}。")
    lines += ["", "## C3–C4 训练与开发集评价", ""]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    fitted = False
    for method in ("full", "numeric"):
        path = ROOT / "training" / f"{method}.json"
        if not path.exists():
            continue
        fitted = True
        run = json.loads(path.read_text())
        offline = json.loads(path.with_name(f"{method}_offline.json").read_text())
        lines.append(f"- {method} 最佳开发集步数 {run['best_step']}，"
                     f"Reward MAE {offline['reward_mae']:.3f}，MSE {offline['reward_mse']:.3f}。")
        for phase in ("anchor", "specialist"):
            for policy in (method, "fixed", "random", "oracle", "restricted_oracle", "reward_oracle"):
                row = offline["decisions"].get(f"{phase}/{policy}")
                if row:
                    lines.append(f"- {method} {phase}/{policy} 奖励 {row['reward']:.3f}，"
                                 f"效用 {row['utility']:.3f}，regret {row['regret']:.3f}。")
        for index, axis in enumerate(axes):
            for field in ("train_loss", "dev_loss"):
                points = [(r["step"], r[field][index]) for r in run["history"] if field in r]
                axis.plot(*zip(*points), label=f"{method} {field}", alpha=.7)
            axis.set_title(("Reward Huber", "Cost Huber")[index])
            axis.set_xlabel("Optimizer step")
    if fitted:
        for axis in axes:
            axis.legend()
        figure.tight_layout()
        figure.savefig(ROOT / "controller_losses.png", dpi=150)
    else:
        lines.append("等待完整采集和审计后训练。")
    plt.close(figure)
    lines += ["", "## C5 独立闭环", ""]
    closed = defaultdict(list)
    for path in (ROOT / "closed").rglob("*.json"):
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        closed[row["mode"]].append(row)
    for mode, rows in sorted(closed.items()):
        actions = Counter(item["selected_action"].split("|")[0] for row in rows for item in row["trace"]["decisions"])
        lines.append(f"- {mode} {len(rows)} 题，平均奖励 "
                     f"{statistics.mean(row['score']['reward'] for row in rows):.3f}，"
                     f"平均耗时 {statistics.mean(row['ms'] for row in rows):.1f} ms，动作分布 {dict(actions)}。")
    if not closed:
        lines.append("等待开发集选定 Controller checkpoint 后运行。")
    lines += ["", "## 证据与限制", "",
              "逐分支 JSON 保存完整生成轨迹、候选、终止原因与资源明细。"
              "训练目录保存按动作的成本误差、奖励校准和事后 Oracle 数据。"
              "Oracle 基于当前有限采样，固定续推下的离线价值与学习策略部署时的价值可能不同。",
              "回归检查记录见 logs/regression.log。历史 Phase 结果及权重独立保存。"]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
