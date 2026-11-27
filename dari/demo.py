"""Offline protocol demonstration, deliberately not an accuracy experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path

from .backends import ScriptedBackend
from .costs import ModelProfile
from .evaluation import ExactMatch
from .runtime import Decision, Runtime, RuntimeConfig
from .schema import Kind, Task


def build_demo(config_path: str | Path | None = None) -> tuple[Runtime, Task]:
    raw = Path(config_path).read_text(encoding="utf-8") if config_path else files("dari").joinpath("defaults.json").read_text(encoding="utf-8")
    config = json.loads(raw)
    profiles = {p["model_id"]: ModelProfile(**p) for p in config["profiles"]}
    if not all(p.synthetic for p in profiles.values()):
        raise ValueError("Scripted demo requires explicitly synthetic profiles")
    first = dict(step="Assume sqrt(x*x)=x, hence -1.", refs=["p1"], goal="Check the sign condition.", answer="-1",
                 help_request={"instruction": "Derive the principal square root using the sign constraint.",
                               "target": "current_step", "required_refs": ["p1"]})
    corrected = dict(step="The principal square root is |x|=1.", refs=["p1"], goal=None, answer="1")
    help_obj = dict(work="Private exploration of sign cases; not sent to Anchor.",
        draft="At x=-1, the principal square root is nonnegative. Use |x| and answer 1.")
    backend = ScriptedBackend({"anchor": [json.dumps(first)] + [json.dumps(corrected)]*16,
        "specialist": [json.dumps(help_obj)]*32})
    rt = Runtime(backend, profiles, RuntimeConfig(**config["runtime"]))
    return rt, Task("synthetic-sign", "Evaluate sqrt(x*x) at x=-1.", {"p1": "x=-1 is negative."})


def finish_policy(state, actions):
    for kind in (Kind.SUBMIT, Kind.ABORT, Kind.FINISH, Kind.CONTINUE):
        found = next((a for a in actions if a.kind == kind), None)
        if found:
            return Decision(found, .25, state.ledger.unit)
    raise ValueError("No fixture continuation")


def run_demo(output_dir: Path | None = None, expand: bool = False,
             oneshot: bool = False, config_path: Path | None = None) -> dict:
    rt, task = build_demo(config_path)

    def policy(state, actions):
        if state.help:
            wanted = Kind.EXPAND if expand and state.help.stage == 1 and state.help.action.quota == "short" else Kind.SUBMIT
            action = next((a for a in actions if a.kind == wanted), None)
            if action is None:
                action = next(a for a in actions if a.kind == Kind.ABORT)
        elif state.help_count == 0:
            action = next(a for a in actions if a.kind == Kind.HELP and a.target == "s1"
                          and a.width == "narrow"
                          and a.quota == ("oneshot" if oneshot else "short"))
        else:
            action = next(a for a in actions if a.kind == Kind.FINISH)
        return Decision(action, .25, state.ledger.unit)

    result = rt.run(task, policy)
    summary = {"notice": "Synthetic protocol demonstration; not evidence for paper performance or efficiency.",
        "synthetic": True, "answer": result.candidate, "terminal": result.terminal,
        "termination_reason": result.termination_reason, "cost": result.ledger.total,
        "cost_unit": result.ledger.unit, "tokens_by_model": result.ledger.tokens_by_model(),
        "retained_help_ids": list(result.artifacts), "controller_cost_unmeasured": result.ledger.unmeasured_overheads}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir/"trace.json").write_text(json.dumps([asdict(e) for e in result.events], indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def collect_demo_targets(path: str | Path) -> None:
    from .collection import collect_branches, write_targets
    rt, task = build_demo()
    initial = rt.initial(task)
    selected = [a for a in rt.actions(initial) if a.kind in (Kind.FINISH, Kind.CONTINUE) or
                (a.kind == Kind.HELP and a.target == "s1" and a.width == "narrow" and a.quota == "short")]
    common = dict(policy=finish_policy, evaluator=ExactMatch(), gold="1", repeats=2,
                  snapshot_id="scripted-A-S-v1-finish-policy", anchor_dim=12, specialist_dim=9)
    rows = collect_branches(rt, initial, selected, **common)
    help_action = next(a for a in selected if a.kind == Kind.HELP)
    helped = rt.step(initial, help_action)
    rows.extend(collect_branches(rt, helped, rt.actions(helped), **common))
    write_targets(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collect", type=Path, help="Write synthetic controller JSONL targets")
    parser.add_argument("--config", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--expand", action="store_true")
    modes.add_argument("--oneshot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, args.expand, args.oneshot, args.config), indent=2, ensure_ascii=False))
    if args.collect:
        collect_demo_targets(args.collect)
        print(f"Synthetic targets written to {args.collect}")


if __name__ == "__main__":
    main()
