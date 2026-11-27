"""Frozen difficult-task selection and paired DARI runs."""
import argparse
from dataclasses import asdict, replace
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random

import torch

from .backends import GenerationRequest
from .configuration import build_runtime
from .phase2b_data import normalized_hash
from .phase2b_evaluation import EVALUATOR_VERSION, score
from .phase3a import write_json
from .phase3a_cost import MeasuredBackend
from .phase3b import ROOT as PHASE
from .pilot import eligible, finish_or_continue, metrics, read_rows, state_id, trace
from .runtime import Runtime
from .schema import Action, Kind, Task

ROOT = PHASE / "challenging_benchmarks"
SEED = 20261201


def prepare():
    import pyarrow.parquet as pq

    prior = [row for path in PHASE.parent.rglob("samples.jsonl") if ROOT not in path.parents
             for row in read_rows(path)]
    for name in ("train", "validation"):
        prior.extend(read_rows(PHASE.parent / "phase2b" / (name + ".jsonl")))
    excluded = {normalized_hash(row["question"]) for row in prior if "question" in row}
    randomizer = random.Random(SEED)
    math = Path("data/phase2a/math500.jsonl")
    hotpot = Path("data/phase2a/hotpot_validation.parquet")
    selected = {}
    for level in (4, 5):
        pool = [dict(task_id="math500:" + row["unique_id"], question=row["problem"],
                     gold=row["answer"], sources={}, level=level)
                for row in read_rows(math) if row["level"] == level
                and normalized_hash(row["problem"]) not in excluded]
        selected.setdefault("math500", []).extend(randomizer.sample(pool, 10))
    pool = []
    for row in pq.read_table(hotpot).to_pylist():
        if row["type"] != "bridge" or row["level"] != "hard" or normalized_hash(row["question"]) in excluded:
            continue
        context = row["context"]
        pairs = zip(context["title"], context["sentences"])
        pool.append(dict(task_id="hotpotqa:" + row["id"], question=row["question"], gold=row["answer"],
                         sources={f"p{i}": title + "\n" + " ".join(lines)
                                  for i, (title, lines) in enumerate(pairs, 1)}))
    selected["hotpotqa"] = randomizer.sample(pool, 10)
    humaneval = Path("data/challenging/humanevalplus-v0.1.10.jsonl.gz")
    if humaneval.is_file():
        with gzip.open(humaneval, "rt", encoding="utf-8") as stream:
            coding_pool = [json.loads(line) for line in stream]
        selected["humaneval"] = [
            dict(task_id=row["task_id"], question=row["prompt"], sources={},
                 entry_point=row["entry_point"])
            for row in randomizer.sample(
                [row for row in coding_pool if normalized_hash(row["prompt"]) not in excluded], 30)
        ]
    livecodebench = Path("data/challenging/livecodebench-test6.jsonl")
    if livecodebench.is_file():
        selected["livecodebench"] = []
        rows = read_rows(livecodebench)
        for difficulty, count in (("easy", 3), ("medium", 14), ("hard", 3)):
            candidates = [row for row in rows if row["difficulty"].lower() == difficulty
                          and normalized_hash(row["question_content"]) not in excluded]
            selected["livecodebench"].extend(
                dict(task_id="LiveCodeBench/" + str(row["question_id"]),
                     question=row["question_content"], difficulty=difficulty,
                     sources={"starter": row["starter_code"]} if row.get("starter_code") else {})
                for row in randomizer.sample(candidates, count))
    for dataset, rows in selected.items():
        for row in rows:
            digest = normalized_hash(row["question"])
            if digest in excluded:
                raise ValueError("Duplicate selected question")
            excluded.add(digest)
        source = {"math500": math, "hotpotqa": hotpot, "humaneval": humaneval,
                  "livecodebench": livecodebench}[dataset]
        write_json(ROOT / dataset / "samples.json", {"seed": SEED,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source": str(source),
            "rule": {"math500": "10 each at levels 4 and 5", "hotpotqa": "hard bridge questions",
                     "humaneval": "30 samples from official EvalPlus v0.1.10",
                     "livecodebench": "test6.jsonl at 0fe84c3912ea0c4d4a78037083943e8f0c4dd505, 3 easy, 14 medium, 3 hard"}[dataset],
            "tasks": rows})


def run(dataset, limit, two_devices=False, shard_specialist=False, shared_base=False):
    directory = ROOT / dataset
    rows = json.loads((directory / "samples.json").read_text())["tasks"]
    if all((directory / "results" / f"{index:02d}.json").is_file()
           for index in range(len(rows[:limit]))):
        return
    config = json.loads((PHASE.parent / "phase3a/manifest.json").read_text())["config"]
    coding = dataset in {"humaneval", "livecodebench"}

    def evaluate(answer, row):
        return {"reward": None, "status": "pending_isolated_evaluation"} if coding else score(dataset, answer, row["gold"])

    for layer in range(18, 21):
        config["backend"]["models"]["S"]["device_map"][f"model.layers.{layer}"] = 2
    if two_devices:
        for device, model in enumerate(("A", "S")):
            config["backend"]["models"][model]["device_map"] = {"": device}
        if shard_specialist:
            placement = {key: value - 2 for key, value in
                         json.loads((PHASE.parent / "phase3a/manifest.json").read_text())
                         ["config"]["backend"]["models"]["S"]["device_map"].items()}
            config["backend"]["models"]["S"]["device_map"] = placement
    config["runtime"].update(anchor_output=1024, max_anchor_steps=4, max_specialist_stages=1,
                             stage_output_budgets=[1024], draft_limit=768, experiment_mode="real_debug")
    if shared_base:
        models = config["backend"]["models"]
        for spec in models.values():
            spec["device_map"] = {"": 0}
        sampling = {"temperature", "top_p"}
        if config.get("model_adapters") or any(spec.get("adapter") for spec in models.values()):
            raise ValueError("Shared base requires adapter-free models")
        if any({k: v for k, v in spec.items() if k not in sampling} !=
               {k: v for k, v in models["A"].items() if k not in sampling}
               for spec in models.values()):
            raise ValueError("Shared base requires identical model and prompt codec configurations")
    manifest = {"config": config, "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "seed": SEED, "evaluator": "pending_official_code_evaluation" if coding else EVALUATOR_VERSION}
    manifest_path = directory / ("manifest_two_devices.json" if two_devices else "manifest_sharded.json")
    if shard_specialist:
        manifest_path = directory / "manifest_two_devices_shared.json"
    if shared_base:
        manifest_path = directory / "manifest_shared_base.json"
        manifest["shared_frozen_base"] = True
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Benchmark deployment changed")
    write_json(manifest_path, manifest)
    if shared_base:
        from .configuration import RecordingBackend
        from .costs import ModelProfile
        from .hf_backend import HuggingFaceBackend
        from .runtime import RuntimeConfig

        loaded = HuggingFaceBackend.from_specs({"A": models["A"]})
        anchor = loaded.handles["A"]
        specialist = replace(anchor, temperature=models["S"].get("temperature", 0.),
                             top_p=models["S"].get("top_p", 1.))
        shared = HuggingFaceBackend({"A": anchor, "S": specialist}, share_frozen_base=True)
        profiles = {p["model_id"]: ModelProfile(**p) for p in config["profiles"]}
        base = Runtime(RecordingBackend(shared), profiles, RuntimeConfig(**config["runtime"]))
    else:
        base = build_runtime(config)
    backend = MeasuredBackend(base.backend)
    runtime = Runtime(backend, base.profiles, base.config)
    for index, row in enumerate(rows[:limit]):
        seed = SEED + index
        task = Task(row["task_id"], row["question"], row["sources"])
        path = directory / "results" / f"{index:02d}.json"
        if path.exists():
            continue
        print(dataset, index, task.task_id, flush=True)
        instruction = ("Follow the problem's input and output contract. Return the entire executable Python program "
                       "inside one Python code fence.\n" if coding
                       else "Solve the task step by step. End with FINAL: followed only by the final answer.\n")
        prompt = (instruction
                  + json.dumps({"task": task.question, "sources": task.sources}, ensure_ascii=False))
        result = backend.generate(GenerationRequest("A", "anchor", prompt, 4096, 0, seed=seed))
        data = {"task_id": task.task_id, "seed": seed, "deployment_manifest": manifest_path.name, "standard": {
            "score": evaluate(result.text, row), "prompt": prompt, "answer": result.text,
            "usage": asdict(result.usage), "truncated": result.usage.output_tokens >= 4096}}
        prefix = runtime.initial(task, seed)
        while runtime.actions(prefix) and not eligible(runtime, prefix, "narrow") and prefix.candidate is None:
            prefix = runtime.decide_step(prefix, finish_or_continue)
        data["prefix_id"] = state_id(prefix)
        data["eligible"] = eligible(runtime, prefix, "narrow")
        data["request"] = asdict(prefix.help_request) if prefix.help_request else None
        data["prefix"] = trace(prefix)

        def save(name, terminal):
            data[name] = {"score": evaluate(terminal.candidate, row),
                          "answer": terminal.candidate, "metrics": metrics(terminal),
                          "ms": terminal.ledger.total, "trace": trace(terminal)}

        anchor = runtime.run_from(prefix, finish_or_continue)
        save("anchor_only", anchor)
        if data["eligible"]:
            repair = runtime.step(prefix, Action(Kind.REPAIR))
            save("self_repair", runtime.run_from(repair, finish_or_continue))
            extended = runtime.step(repair, Action(Kind.REPAIR)) if Action(Kind.REPAIR) in runtime.actions(repair) else repair
            save("self_repair_extended", runtime.run_from(extended, finish_or_continue))
            help_action = next(a for a in runtime.actions(prefix) if a.kind == Kind.HELP and a.width == "narrow" and a.quota == "short")
            for name, internal in (("help", False), ("internal_restart", True)):
                branch = runtime
                if internal:
                    branch = Runtime(backend, base.profiles, replace(runtime.config,
                        specialist_model="A", specialist_context="independent"))
                helped = branch.step(prefix, help_action)
                action = Action(Kind.SUBMIT) if Action(Kind.SUBMIT) in branch.actions(helped) else Action(Kind.ABORT)
                received = branch.step(helped, action) if action in branch.actions(helped) else helped
                save(name, branch.run_from(received, finish_or_continue))
        write_json(path, data)


def summarize():
    summary = {}
    for selection in sorted(ROOT.glob("*/samples.json")):
        directory = selection.parent
        rows = [json.loads(path.read_text()) for path in sorted((directory / "results").glob("*.json"))]
        results = {"selected": len(json.loads(selection.read_text())["tasks"]), "completed": len(rows),
                   "eligible": sum(row["eligible"] for row in rows), "methods": {}}
        methods = ("standard", "anchor_only", "self_repair", "self_repair_extended", "internal_restart", "help")
        for method in methods:
            scores = [row[method]["score"]["reward"] for row in rows if method in row]
            scored = [value for value in scores if value is not None]
            results["methods"][method] = {"generated": len(scores), "scored": len(scored),
                                          "mean_reward": sum(scored) / len(scored) if scored else None}
        summary[directory.name] = results
    write_json(ROOT / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "summarize", "math500", "hotpotqa", "humaneval", "livecodebench"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--two-devices", action="store_true", help="Place each frozen model on its own GPU")
    parser.add_argument("--shard-specialist", action="store_true", help="Shard Specialist across both visible GPUs")
    parser.add_argument("--shared-base", action="store_true", help="Reuse one frozen BF16 model across roles")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "summarize":
        summarize()
    else:
        directory = ROOT / args.stage
        with (directory / ".worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run(args.stage, args.limit, args.two_devices or args.shard_specialist,
                args.shard_specialist, args.shared_base)
