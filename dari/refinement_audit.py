"""Offline diagnosis of immutable Phase 2A traces using the original tokenizer."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics

from .pilot import read_rows


def diagnose():
    from transformers import AutoTokenizer
    original=Path("outputs/phase2a")
    root=Path("outputs/phase2a_refinement")
    root.mkdir(exist_ok=True)
    config=json.loads(Path("configs/phase2a_qwen3_8b.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained(config["backend"]["models"]["S"]["name_or_path"],local_files_only=True,trust_remote_code=False)
    count=lambda text:len(tokenizer.encode(text,add_special_tokens=False))
    rows=read_rows(original/"e1/results.jsonl")
    help_rows={row["task_id"]:row for row in rows if row["method"]=="help_submit"}
    lengths=[]
    for row in read_rows(original/"e1/traces.jsonl"):
        if row["method"]!="help_submit":
            continue
        event=next(event for event in row["events"] if event["role"]=="specialist")
        obj=json.loads(event["output"])
        draft=obj["draft"]
        compact=json.dumps(draft,ensure_ascii=False,separators=(",",":"))
        values="\n".join(draft.values())
        record={"task_id":row["task_id"],"prefix_id":row["prefix_id"],"work":obj["work"],
            "work_tokens":count(obj["work"]),"draft_compact_tokens":count(compact),
            "draft_text_tokens":count(values),"json_overhead_difference":count(compact)-count(values),
            "field_tokens":{key:count(value) for key,value in draft.items()},
            "output_tokens":event["usage"]["output_tokens"],"complete_json":True,
            "generation_hit_cap":event["usage"]["output_tokens"]>=512,
            "rejected_after_parsing":event["error"]=="Draft payload exceeds quota",
            "placeholder_work":obj["work"].strip().lower()=="private local analysis",
            "request":help_rows[row["task_id"]]["dynamic_request"],
            "original_question_present":True,"raw_draft":draft,
            "prompt":event["prompt"]}
        question=next(sample["question"] for sample in read_rows(original/"samples.jsonl") if sample["task_id"]==row["task_id"])
        record["original_question_present"]=question in event["prompt"] or json.dumps(question,ensure_ascii=False)[1:-1] in event["prompt"]
        lengths.append(record)
    e0=read_rows(original/"e0/results.jsonl")
    e0traces={(row["task_id"],row["method"]):row for row in read_rows(original/"e0/traces.jsonl")}
    standard={row["task_id"]:row for row in e0 if row["method"]=="standard_cot"}
    failures=[]
    for row in e0:
        if row["method"]=="standard_cot" or standard[row["task_id"]]["reward"]!=1 or row["reward"]==1:
            continue
        events=e0traces[(row["task_id"],row["method"])]["events"]
        steps=e0traces[(row["task_id"],row["method"])]["steps"]
        if row["termination_reason"]=="anchor_format":
            category="format_failure"
        elif row["termination_reason"]=="step_limit":
            category="step_limit"
        elif row["evaluator_status"]=="unparsed_answer":
            category="answer_extraction_review"
        else:
            category="reasoning_or_premature_answer_requires_review"
        failures.append({"task_id":row["task_id"],"method":row["method"],"category":category,
            "answer":row["answer"],"steps":steps,"errors":[event["error"] for event in events if not event["valid"]],
            "output_cap_hits":sum(event["usage"]["output_tokens"]>=512 for event in events),
            "last_output":events[-1]["output"]})
    summary={"n":len(lengths),"draft_tokens":{"min":min(row["draft_compact_tokens"] for row in lengths),
        "median":statistics.median(row["draft_compact_tokens"] for row in lengths),"max":max(row["draft_compact_tokens"] for row in lengths)},
        "offline_budget_sensitivity":{str(limit):sum(row["draft_compact_tokens"]<=limit for row in lengths) for limit in (192,256,384,512)},
        "placeholder_work":sum(row["placeholder_work"] for row in lengths),
        "original_question_present":sum(row["original_question_present"] for row in lengths),
        "parse_complete":sum(row["complete_json"] for row in lengths),
        "rejected_post_parse":sum(row["rejected_after_parsing"] for row in lengths),
        "fallback_sources":dict(Counter("null" if row["request"]["fallback"] else "anchor_request" for row in lengths)),
        "anchor_failure_categories":dict(Counter(row["category"] for row in failures)),
        "confirmed_root_causes":["Prompt uses literal private local analysis as pseudo-schema value; all copied work fields are model text, not runtime defaults",
            "Old quota measures compact serialized whole draft including target, field keys and repeated advisory report",
            "Original question is present; egg-sale omission is Specialist failure to use original question, not question truncation",
            "Fallback at saved checkpoints originates from valid Anchor help_request=null, not parser substitution",
            "Anchor history repeats old goals, requests, candidates and invalidate flags, encouraging field copying; valid math reasoning also fails on JSON escaping"],
        "not_established":["All wrong valid answers are premature completion", "All overlong drafts are useful", "Missing document evidence never contributes"]}
    (root/"root_cause.json").write_text(json.dumps(summary,indent=2))
    (root/"r0_original_outputs.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in lengths))
    (root/"anchor_failure_audit.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in failures))
    rng=random.Random(20261125)
    ids=sorted(row["prefix_id"] for row in lengths)
    selected=rng.sample(ids,12)
    plan={"seed":20261125,"r0_prefix_ids":selected,"r2_pilot_prefix_ids":selected,
        "r2_all_prefix_ids":ids,"r1_task_ids":[row["task_id"] for row in read_rows(original/"samples.jsonl") if row["pilot"]],
        "selection":"random without replacement before refined generation; R1 uses original per-dataset pilot 8"}
    path=root/"selection.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError("Selection differs")
    path.write_text(json.dumps(plan,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    diagnose()
