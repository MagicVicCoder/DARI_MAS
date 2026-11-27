"""Verify completed refinement condition coverage and immutable prefix identity."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from .pilot import BRANCHES, METHODS, read_rows, state_id
from .refinement import ROOT, OLD


def verify():
    plan = json.loads((ROOT/"selection.json").read_text())
    old = read_rows(OLD/"samples.jsonl")
    fresh = read_rows(ROOT/"r3/samples.jsonl")
    assert len(fresh) == len({row["task_id"] for row in fresh}) == 24
    assert not ({row["task_id"] for row in fresh} & {row["task_id"] for row in old})
    assert Counter(row["dataset"] for row in fresh) == {"gsm8k":8,"math500":8,"hotpotqa":8}
    checked = {}
    for stage, subdir, expected in [("r1","e0",48),("r2","e1",144),("r3","e0",72),("r3_branches","e1",56)]:
        rows = read_rows(ROOT/stage/subdir/"results.jsonl")
        assert len(rows) == expected and all(row["status"] == "completed" for row in rows)
        keys = {(row["task_id"],row["method"]) for row in rows}
        assert len(keys) == expected
        grouped = {}
        for row in rows:
            grouped.setdefault(row["task_id"],set()).add(row["method"])
        methods = set(BRANCHES if subdir == "e1" else METHODS if stage == "r3" else METHODS[1:])
        assert all(value == methods for value in grouped.values())
        if stage == "r1":
            assert set(grouped) == set(plan["r1_task_ids"])
        if subdir == "e1":
            assert all(row["prefix_unchanged"] for row in rows)
            ids = {row["prefix_id"] for row in rows}
            directory = OLD/"e1/prefixes" if stage == "r2" else ROOT/"r3/e1/prefixes"
            for prefix_id in ids:
                state = torch.load(directory/(prefix_id+".pt"),weights_only=False,map_location="cpu")
                assert state_id(state) == prefix_id
            if stage == "r2":
                assert ids == set(plan["r2_all_prefix_ids"])
            else:
                selected = {row["prefix_id"] for row in read_rows(ROOT/"r3/e0/results.jsonl") if row.get("prefix_id")}
                assert ids == selected
                coverage = read_rows(ROOT/"r3_branches/e1/coverage.jsonl")
                assert len(coverage) == 10
            checked[stage] = {"conditions":expected,"prefixes":len(ids),"prefix_hashes_verified":True}
        else:
            checked[stage] = {"conditions":expected,"tasks":len(grouped)}
        manifest = json.loads((ROOT/stage/"manifest.json").read_text())
        assert all(row["experiment_id"] == manifest["experiment_id"] for row in rows)
        for name, digest in manifest["code"].items():
            assert hashlib.sha256(Path("dari",name).read_bytes()).hexdigest() == digest, name
        checked[stage]["generation_code_hashes_verified"] = True
    replay = read_rows(ROOT/"r0/replay384_results.jsonl")
    assert len(replay) == len({row["prefix_id"] for row in replay}) == 12
    assert {row["prefix_id"] for row in replay} == set(plan["r0_prefix_ids"])
    result = {"passed":True,"fresh_tasks_disjoint":True,"stages":checked,"r0_replays":12,
              "code_hash_note":"Exact generation source hashes match every recorded manifest; later analysis has separate Git history."}
    (ROOT/"integrity_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    verify()
