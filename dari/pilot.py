"""Resumable E0/E1 real-model development experiments on the existing Runtime."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from .backends import GenerationRequest
from .configuration import build_runtime
from .context import build_pack
from .controller import encode_state_actions
from .identity import fingerprint
from .pilot_evaluation import EVALUATOR_VERSION, score
from .runtime import Decision, Runtime, CONTINUE_INSTRUCTION
from .schema import Action, Kind, Task

METHODS = ("standard_cot", "stepwise_cot", "dari_anchor_only")
BRANCHES = ("continue", "self_repair", "self_repair_extended", "help_submit")


def append(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(body, ensure_ascii=False, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finish_or_continue(state, legal):
    action = next((action for action in legal if action.kind == Kind.FINISH), None)
    action = action or next((action for action in legal if action.kind == Kind.CONTINUE), None)
    if action is None:
        raise ValueError("No legal Anchor-only continuation")
    return Decision(action)


def runtime_for(runtime, **changes):
    return Runtime(runtime.backend, runtime.profiles, replace(runtime.config, **changes))


def help_action(runtime, prefix, width):
    return next((action for action in runtime.actions(prefix) if action.kind == Kind.HELP and
                 action.width == width and action.quota == "short"), None)


def eligible(runtime, state, width):
    if state.terminal or state.candidate is not None or not state.steps or not state.help_request:
        return False
    legal = runtime.actions(state)
    return Action(Kind.CONTINUE) in legal and Action(Kind.REPAIR) in legal and help_action(runtime, state, width) is not None


def state_id(state):
    return fingerprint(asdict(state))


def resources(state, prefix=None):
    entries = state.ledger.entries[len(prefix.ledger.entries):] if prefix else state.ledger.entries
    result = {}
    for usage, _ in entries:
        counts = result.setdefault(usage.model_id, {"input":0,"output":0,"calls":0,"cached_input":0})
        for key, value in (("input", usage.input_tokens), ("output", usage.output_tokens), ("cached_input",usage.cached_input_tokens), ("calls",1)):
            counts[key] += value
    return result


def metrics(state, prefix=None):
    events = state.events[len(prefix.events):] if prefix else state.events
    requests = [step.record.help_request.instruction for step in state.steps if step.record.help_request]
    canonical = [" ".join(request.lower().split()) for request in requests]
    repeated = sum(request in canonical[:index] for index, request in enumerate(canonical))
    similar = sum(any(SequenceMatcher(None, request, other).ratio() >= .9 for other in canonical[:index]) for index,request in enumerate(canonical))
    anchor_events = [event for event in events if event.role == "anchor"]
    return {"tokens_by_model":resources(state,prefix), "termination_reason":state.termination_reason,
        "format_failures":sum(not event.valid for event in events),
        "first_format_failure":bool(anchor_events and not anchor_events[0].valid),
        "invalid_call_rate":sum(not event.valid for event in events)/max(1,len(events)),
        "no_answer":state.candidate is None, "reasoning_steps":len(state.steps)-(len(prefix.steps) if prefix else 0),
        "request_count":len(requests), "request_repeated":repeated,"request_similar":similar,
        "cost_complete":False,"budget_exceeded":state.ledger.total > state.budget}


def trace(state, prefix=None):
    return {"events":[asdict(event) for event in (state.events[len(prefix.events):] if prefix else state.events)],
            "steps":[asdict(step) for step in state.steps], "decisions":state.decision_log,
            "assistance":state.assistance_log, "retained_assistance":state.artifacts,
            "help_request":asdict(state.help_request) if state.help_request else None}


class Experiment:
    def __init__(self, config_path):
        self.config = json.loads(Path(config_path).read_text())
        self.root = Path(self.config["output_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples = read_rows(Path(self.config["sample_file"]))
        source = {name:hashlib.sha256(Path("dari",name).read_bytes()).hexdigest() for name in
                  ("pilot.py","pilot_data.py","pilot_evaluation.py","runtime.py","context.py","hf_backend.py","schema.py")}
        identity = {"config":self.config,"samples":fingerprint(self.samples),"code":source,
                    "evaluator":EVALUATOR_VERSION}
        self.identity = fingerprint(identity)
        manifest_path = self.root/"manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["experiment_id"] != self.identity:
            raise ValueError("Resume rejected: model/prompt/protocol/evaluator/samples changed; select a new output_dir")
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps({**identity,"experiment_id":self.identity,
                "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                "software":{"torch":torch.__version__}, "cost_complete":False,
                "checkpoint_rule":self.config["checkpoint_selection"],
                "branch_protocol":{"continue":"one normal Anchor call", "self_repair":"one local-task Anchor call",
                    "self_repair_extended":"one local-task Anchor call plus one integration/refinement call",
                    "help_submit":"one Specialist call plus one Anchor integration call; invalid draft aborts",
                    "common_continuation":"up to 8 new valid Anchor steps after intervention, stop at candidate; retries billed"}},indent=2))
        self.runtime = build_runtime(deepcopy(self.config))
        self.backend = self.runtime.backend.backend
        self.dimensions = {"anchor_dim":self.backend.handles["A"].hidden_dim,
                           "specialist_dim":self.backend.handles["S"].hidden_dim}
        (self.root/"model_manifest.json").write_text(json.dumps(self.backend.deployment_identity,indent=2))

    def save_prefix(self, state, row):
        prefix_id = state_id(state)
        directory = self.root/"e1/prefixes"
        directory.mkdir(parents=True,exist_ok=True)
        path = directory/(prefix_id+".pt")
        if not path.exists():
            torch.save(state,path)
            actions = self.runtime.actions(state)
            obs, encoded = encode_state_actions(self.runtime,state,actions,**self.dimensions)
            body = {"prefix_id":prefix_id,"task_id":row["task_id"],"state":asdict(state),
                "observation":{key:value.tolist() for key,value in obs.items()},
                "legal_actions":[action.key for action in actions],
                "action_features":{key:value.tolist() for key,value in encoded.items()},
                "experiment_id":self.identity,"model_snapshot":self.backend.deployment_identity,
                "downstream_policy":"fixed_finish_else_continue_v1"}
            path.with_suffix(".json").write_text(json.dumps(body,ensure_ascii=False))
        return prefix_id

    def e0(self, rows):
        result_path=self.root/"e0/results.jsonl"
        done={(row["task_id"],row["method"]) for row in read_rows(result_path) if row["status"]=="completed"}
        for row in rows:
            task=Task(row["task_id"],row["question"],row["sources"])
            seed=self.config["seed"]+row["row_index"]
            for method in METHODS:
                if (task.task_id,method) in done:
                    continue
                print(f'E0 {task.task_id} {method}',flush=True)
                started=time.perf_counter()
                base={"task_id":task.task_id,"dataset":row["dataset"],"method":method,"experiment_id":self.identity,"seed":seed}
                try:
                    prefix_id=None
                    if method=="standard_cot":
                        prompt=("Solve the problem step by step using the provided evidence and constraints. "
                                "End with a separate line FINAL: followed by only the concise final answer. "
                                "For mathematics use standard LaTeX.\n"+json.dumps({"task":task.question,"sources":task.sources},ensure_ascii=False))
                        request=GenerationRequest("A","anchor",prompt,self.config["standard_output"],0,seed=seed)
                        generation=self.runtime.backend.generate(request)
                        from .costs import CostLedger
                        from .runtime import EpisodeState
                        from .schema import CallEvent
                        state=EpisodeState(task,CostLedger(self.runtime.profiles),self.runtime.config.budget)
                        state.ledger.charge(generation.usage)
                        from .pilot_evaluation import final_answer
                        state.candidate=final_answer(generation.text)
                        state.termination_reason="output_limit" if generation.usage.output_tokens>=request.max_new_tokens else "completed"
                        state.events.append(CallEvent("anchor",prompt,generation.text,True,usage=asdict(generation.usage),latency_ms=generation.latency_ms,raw_output=generation.raw_text))
                    else:
                        runtime=runtime_for(self.runtime,generate_help_request=method=="dari_anchor_only")
                        state=runtime.initial(task,seed)
                        while not state.terminal:
                            if method=="dari_anchor_only" and prefix_id is None and eligible(runtime,state,self.config["context_width"]):
                                prefix_id=self.save_prefix(state,row)
                            legal=runtime.actions(state)
                            if not legal:
                                runtime._terminate(state,"no_feasible_action")
                                break
                            state=runtime.decide_step(state,finish_or_continue)
                    record={**base,"status":"completed",**metrics(state),**score(row["dataset"],state.candidate,row["gold"]),
                        "answer":state.candidate,"prefix_id":prefix_id,"e2e_ms":(time.perf_counter()-started)*1000,
                        "configured_output_budget":self.config["standard_output"] if method=="standard_cot" else self.runtime.config.max_anchor_steps*self.runtime.config.anchor_output,
                        "shared_prefix_resources":None}
                    append(self.root/"e0/traces.jsonl",{**base,**trace(state)})
                    append(result_path,record)
                except Exception as error:
                    append(result_path,{**base,"status":"failed","error":repr(error),"e2e_ms":(time.perf_counter()-started)*1000})
                    raise

    def branch(self,prefix,method,seed):
        shared_hash=state_id(prefix)
        state=deepcopy(prefix)
        state.seed=seed
        # Initial interventions have one/two calls; common continuation gets the
        # same number of NEW steps after each completed intervention, not a shared
        # absolute total-step cap that would penalize extra integration calls.
        runtime=runtime_for(self.runtime,max_anchor_steps=len(prefix.steps)+2+self.config["continuation_steps"])
        initial_request=deepcopy(state.help_request)
        submitted=False
        selected_ids=[]
        if method in {"continue","self_repair","self_repair_extended"}:
            action=Action(Kind.CONTINUE if method=="continue" else Kind.REPAIR)
            state=runtime.step(state,action)
            if method=="self_repair_extended" and not state.terminal:
                runtime._anchor(state,"Integrate and refine your reasoning for the following original local task, using all your new work. "
                                "Check the result and advance the global solution: "+initial_request.instruction)
        elif method=="help_submit":
            action=help_action(runtime,state,self.config["context_width"])
            if action is None:
                raise ValueError("Preselected checkpoint lacks Help action")
            selected_ids=list(build_pack(state,action.target,action.width).span_ids)
            state=runtime.step(state,action)
            action=Action(Kind.SUBMIT) if Action(Kind.SUBMIT) in runtime.actions(state) else Action(Kind.ABORT)
            submitted=action.kind==Kind.SUBMIT
            state=runtime.step(state,action)
        else:
            raise ValueError(method)
        intervention_resources=resources(state,prefix)
        intervention_steps=len(state.steps)-len(prefix.steps)
        runtime=runtime_for(runtime,max_anchor_steps=len(state.steps)+self.config["continuation_steps"])
        state=runtime.run_from(state,finish_or_continue)
        if state_id(prefix)!=shared_hash:
            raise RuntimeError("Counterfactual branch mutated shared prefix")
        return state,{"intervention_resources":intervention_resources,"intervention_steps":intervention_steps,
            "common_continuation_step_cap":self.config["continuation_steps"],"submitted":submitted,
            "explicitly_retained":bool(state.artifacts),"selected_context_ids":selected_ids,
            "dynamic_request":asdict(initial_request),"shared_prefix_resources":resources(prefix),
            "logical_end_to_end_resources":resources(state),"prefix_unchanged":True}

    def e1(self, rows, robustness=False):
        e0={row["task_id"]:row for row in read_rows(self.root/"e0/results.jsonl") if row.get("method")=="dari_anchor_only" and row["status"]=="completed"}
        directory=self.root/("robustness" if robustness else "e1")
        results_path=directory/"results.jsonl"
        done={(row["task_id"],row["method"],row.get("repeat",0),row.get("specialist_temperature",.7)) for row in read_rows(results_path) if row["status"]=="completed"}
        if robustness:
            available=[row for row in rows if e0.get(row["task_id"],{}).get("prefix_id")]
            rng=random.Random(self.config["seed"]+997)
            rows=rng.sample(available,min(self.config["robustness_checkpoints"],len(available)))
            plan=[(method,repeat,temp) for temp in (0.,.7) for repeat in range(self.config["rollout_repeats"]) for method in ("self_repair_extended","help_submit")]
        else:
            plan=[(method,0,.7) for method in BRANCHES]
        for row in rows:
            prefix_id=e0.get(row["task_id"],{}).get("prefix_id")
            if not prefix_id:
                if not robustness and not any(r["task_id"]==row["task_id"] for r in read_rows(directory/"coverage.jsonl")):
                    append(directory/"coverage.jsonl",{"task_id":row["task_id"],"status":"skipped","reason":"no_eligible_unfinished_checkpoint"})
                continue
            prefix=torch.load(self.root/"e1/prefixes"/(prefix_id+".pt"),weights_only=False,map_location="cpu")
            if state_id(prefix)!=prefix_id:
                raise ValueError("Saved prefix integrity mismatch")
            for method,repeat,temperature in plan:
                if (row["task_id"],method,repeat,temperature) in done:
                    continue
                print(f'E1 {row["task_id"]} {method} repeat={repeat} S_temp={temperature}',flush=True)
                self.backend.handles["S"].temperature=temperature
                started=time.perf_counter()
                seed=prefix.seed+(repeat+1)*100003 if robustness else prefix.seed
                base={"task_id":row["task_id"],"dataset":row["dataset"],"prefix_id":prefix_id,"method":method,
                    "experiment_id":self.identity,"repeat":repeat,"specialist_temperature":temperature,"seed":seed}
                try:
                    state,meta=self.branch(prefix,method,seed)
                    append(directory/"traces.jsonl",{**base,**trace(state,prefix)})
                    append(results_path,{**base,"status":"completed",**meta,**metrics(state,prefix),
                        **score(row["dataset"],state.candidate,row["gold"]),"answer":state.candidate,
                        "e2e_ms":(time.perf_counter()-started)*1000,
                        "configured_intervention_output_budget":1024 if method in {"self_repair_extended","help_submit"} else 512})
                except Exception as error:
                    append(results_path,{**base,"status":"failed","error":repr(error)})
                    raise
        self.backend.handles["S"].temperature=.7


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/phase2a_qwen3_8b.json")
    parser.add_argument("--stage",choices=("e0","e1","robustness","pilot","expand"),default="pilot")
    parser.add_argument("--per-dataset",type=int)
    args=parser.parse_args()
    experiment=Experiment(args.config)
    size=args.per_dataset or (20 if args.stage in {"expand","robustness"} else 8)
    rows=[row for row in experiment.samples if row["selection_rank"]<size]
    if args.stage in {"e0","pilot","expand"}:
        experiment.e0(rows)
    if args.stage in {"e1","pilot","expand","robustness"}:
        experiment.e1(rows,robustness=args.stage=="robustness")
    manifest=experiment.root/"execution.json"
    manifest.write_text(json.dumps({"last_stage":args.stage,"per_dataset":size,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_allocated_bytes":{str(i):torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())},
        "actual_executed_calls_recorded_in_results":True},indent=2))


if __name__ == "__main__":
    main()
