"""Run frozen DARI inference from a JSON deployment config."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from .configuration import build_runtime, evaluator_for, load_dataset, make_policy


def run_dataset(config: dict, tasks: str | Path, output: str | Path,
                evaluate: bool = False, trace: bool = False) -> list[dict]:
    records = load_dataset(tasks, require_gold=evaluate)
    runtime = build_runtime(config)
    policy = make_policy(runtime, config)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w", encoding="utf-8") as stream:
        for number, record in enumerate(records):
            start = time.perf_counter()
            terminal = runtime.run(record.task, policy, seed=config.get("seed", 7)+number)
            row = {"task_id": record.task.task_id, "answer": terminal.candidate,
                   "termination_reason": terminal.termination_reason,
                   "cost": terminal.ledger.total, "cost_unit": terminal.ledger.unit,
                   "cost_complete": not terminal.ledger.unmeasured_overheads,
                   "tokens_by_model": terminal.ledger.tokens_by_model(),
                   "synthetic": runtime.backend.synthetic,
                   "e2e_ms": (time.perf_counter()-start)*1000,
                   "decisions": terminal.decisions, "help_sessions": terminal.help_count}
            if evaluate:
                row["score"] = evaluator_for(record.metric).score(terminal.candidate, record.gold)
            if trace:
                row["trace"] = [asdict(event) for event in terminal.events]
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = run_dataset(config, args.tasks, args.output, args.evaluate, args.trace)
    print(json.dumps({"examples": len(rows), "output": str(args.output),
                      "synthetic": rows[0]["synthetic"]}, indent=2))


if __name__ == "__main__":
    main()
