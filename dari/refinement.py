"""Frozen refinement experiments reusing the pilot Runtime and branch runner."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .pilot import Experiment, append, read_rows, metrics, trace, state_id, runtime_for, finish_or_continue
from .pilot_data import prepare
from .pilot_evaluation import score
from .schema import Action, Kind
from .runtime import SUBMIT_INSTRUCTION

ROOT = Path("outputs/phase2a_refinement")
OLD = Path("outputs/phase2a")


def setup():
    original = read_rows(OLD/"samples.jsonl")
    prepare(output=ROOT/"r3", seed=20261125, per_dataset=8, exclude=original)
    base = json.loads(Path("configs/phase2a_qwen3_8b.json").read_text())
    base.update(protocol_version="refinement-v3-work-draft-frozen", seed=20261125)
    base["runtime"].update(draft_limit=384)
    for name, devices in (("A", (0, 1)), ("S", (2, 3))):
        spec = base["backend"]["models"][name]
        spec.pop("device", None)
        spec["device_map"] = {"model.embed_tokens": devices[0],
            **{f"model.layers.{i}": devices[0] if i < 18 else devices[1] for i in range(36)},
            "model.norm": devices[1], "lm_head": devices[1]}
    for stage in ("r1", "r2", "r3"):
        config = deepcopy(base)
        config.update(output_dir=str(ROOT/stage), sample_file=str(ROOT/"r3/samples.jsonl") if stage == "r3" else str(OLD/"samples.jsonl"))
        if stage == "r1":
            config["methods"] = ["stepwise_cot", "dari_anchor_only"]
        path = Path(f"configs/refinement_{stage}.json")
        content = json.dumps(config, indent=2)+"\n"
        if path.exists() and path.read_text() != content:
            raise ValueError("Frozen configuration changed")
        path.write_text(content)


def old_prefixes(experiment, selected):
    index = {row["prefix_id"]: row for row in read_rows(OLD/"e1/results.jsonl")}
    samples = {row["task_id"]: row for row in experiment.samples}
    done = {(row["prefix_id"], row["method"]) for row in read_rows(experiment.root/"e1/results.jsonl") if row["status"] == "completed"}
    for prefix_id in selected:
        prefix = torch.load(OLD/"e1/prefixes"/(prefix_id+".pt"), weights_only=False, map_location="cpu")
        if state_id(prefix) != prefix_id:
            raise ValueError("Original prefix changed")
        row = samples[index[prefix_id]["task_id"]]
        for method in ("continue", "self_repair", "self_repair_extended", "help_submit"):
            if (prefix_id, method) in done:
                continue
            print(f'R2 {row["task_id"]} {method}', flush=True)
            started = time.perf_counter()
            state, meta = experiment.branch(prefix, method, prefix.seed)
            base = {"task_id":row["task_id"], "dataset":row["dataset"], "prefix_id":prefix_id,
                    "method":method, "seed":prefix.seed, "experiment_id":experiment.identity}
            append(experiment.root/"e1/traces.jsonl", {**base, **trace(state, prefix)})
            append(experiment.root/"e1/results.jsonl", {**base, "status":"completed", **meta, **metrics(state,prefix),
                **score(row["dataset"],state.candidate,row["gold"]), "answer":state.candidate,
                "e2e_ms":(time.perf_counter()-started)*1000})


def replay_old(experiment, selected):
    records = {row["prefix_id"]: row for row in read_rows(ROOT/"r0_original_outputs.jsonl")}
    samples = {row["task_id"]: row for row in experiment.samples}
    done = {row["prefix_id"] for row in read_rows(ROOT/"r0/replay384_results.jsonl")}
    for prefix_id in selected:
        if prefix_id in done:
            continue
        prefix = torch.load(OLD/"e1/prefixes"/(prefix_id+".pt"), weights_only=False, map_location="cpu")
        original_hash = state_id(prefix)
        record = records[prefix_id]
        row = samples[record["task_id"]]
        state = deepcopy(prefix)
        runtime = runtime_for(experiment.runtime, max_anchor_steps=len(prefix.steps)+1+experiment.config["continuation_steps"])
        payload = json.dumps(record["raw_draft"], ensure_ascii=False, separators=(",",":"))
        if record["draft_compact_tokens"] > 384:
            raise ValueError("Historical payload exceeds selected budget")
        print(f'R0 replay384 {row["task_id"]}', flush=True)
        started = time.perf_counter()
        runtime._anchor(state, SUBMIT_INSTRUCTION, ("h1",payload))
        runtime = runtime_for(runtime,max_anchor_steps=len(state.steps)+experiment.config["continuation_steps"])
        state = runtime.run_from(state, finish_or_continue)
        if state_id(prefix) != original_hash:
            raise ValueError("Replay mutated original prefix")
        base = {"prefix_id":prefix_id,"task_id":row["task_id"],"dataset":row["dataset"],"method":"old_structure_384_replayed",
            "specialist_generation_reused":True,"new_specialist_calls":0,"historical_specialist_output_tokens":record["output_tokens"],
            "historical_specialist_prompt":record["prompt"],"draft_tokens":record["draft_compact_tokens"],
            "downstream":"refined_anchor","prefix_unchanged":True}
        append(ROOT/"r0/replay384_traces.jsonl",{**base,**trace(state,prefix)})
        append(ROOT/"r0/replay384_results.jsonl",{**base,**metrics(state,prefix),
            **score(row["dataset"],state.candidate,row["gold"]),"answer":state.candidate,
            "e2e_ms":(time.perf_counter()-started)*1000})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("setup","r0","r1","r2pilot","r2","r3"))
    args = parser.parse_args()
    if args.stage == "setup":
        setup()
        return
    plan = json.loads((ROOT/"selection.json").read_text())
    stage = args.stage if args.stage in {"r1","r3"} else "r2"
    experiment = Experiment(f"configs/refinement_{stage}.json")
    if args.stage == "r1":
        experiment.e0([row for row in experiment.samples if row["task_id"] in plan["r1_task_ids"]])
    elif args.stage == "r3":
        experiment.e0(experiment.samples)
        experiment.e1(experiment.samples)
    else:
        selected = plan["r2_all_prefix_ids"] if args.stage == "r2" else plan["r2_pilot_prefix_ids"]
        if args.stage == "r0":
            replay_old(experiment, selected)
        old_prefixes(experiment, selected)


if __name__ == "__main__":
    main()
