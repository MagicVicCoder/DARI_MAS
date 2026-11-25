"""Deterministic development samples with original rows and content provenance."""
import hashlib
import json
from pathlib import Path
import random


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root="data/phase2a", output="outputs/phase2a", seed=20261124, per_dataset=20, exclude=()):
    import pyarrow.parquet as pq
    root, output = Path(root), Path(output)
    rng = random.Random(seed)
    files = {"gsm8k": root/"gsm8k.jsonl", "math500": root/"math500.jsonl", "hotpotqa": root/"hotpot_validation.parquet"}
    datasets = {name: [json.loads(line) for line in path.read_text().splitlines()] for name, path in files.items() if name != "hotpotqa"}
    datasets["hotpotqa"] = pq.read_table(files["hotpotqa"]).to_pylist()
    samples = []
    excluded = {(row["dataset"], row["row_index"]) for row in exclude}
    for name, rows in datasets.items():
        if name == "math500":
            # Round robin across subject/difficulty strata, not model behavior.
            groups = {}
            for index, row in enumerate(rows):
                if (name, index) in excluded:
                    continue
                groups.setdefault((row.get("subject", row.get("type", "")), str(row.get("level", ""))), []).append(index)
            for group in groups.values():
                rng.shuffle(group)
            keys = sorted(groups)
            rng.shuffle(keys)
            selected = []
            while len(selected) < per_dataset:
                for key in keys:
                    if groups[key]:
                        selected.append(groups[key].pop())
                        if len(selected) == per_dataset:
                            break
        else:
            selected = rng.sample([i for i in range(len(rows)) if (name, i) not in excluded], per_dataset)
        for rank, index in enumerate(selected):
            row = rows[index]
            sources = {}
            if name == "gsm8k":
                question, gold = row["question"], row["answer"].split("####")[-1].strip()
                sample_id = str(index)
            elif name == "math500":
                question, gold = row["problem"], row["answer"]
                sample_id = str(row.get("unique_id", index))
            else:
                question, gold = row["question"], row["answer"]
                context = row["context"]
                pairs = zip(context["title"], context["sentences"]) if isinstance(context, dict) else context
                sources = {f"p{i+1}": title+"\n"+" ".join(sentences) for i, (title, sentences) in enumerate(pairs)}
                sample_id = row.get("id", row.get("_id", str(index)))
            samples.append({"task_id": name+":"+sample_id, "dataset": name,
                "question": question, "sources": sources, "gold": gold, "original": row,
                "sample_id": sample_id, "row_index": index, "selection_rank": rank,
                "pilot": rank < 8, "split": "validation" if name == "hotpotqa" else "test",
                "dataset_revision": sha256(files[name]), "exploratory_development": True,
                "dataset_source": {"gsm8k": "local AgentPrune copy of GSM8K (content version recorded; upstream revision unknown)",
                                   "math500": "HuggingFaceH4/MATH-500/test.jsonl via hf-mirror.com",
                                   "hotpotqa": "hotpotqa/hotpot_qa/distractor validation via hf-mirror.com"}[name]})
    output.mkdir(parents=True, exist_ok=True)
    path = output/"samples.jsonl"
    text = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in samples)
    if path.exists() and path.read_text() != text:
        raise ValueError("Existing sample manifest differs; do not overwrite an experiment")
    path.write_text(text)
    (output/"data_manifest.json").write_text(json.dumps({"seed": seed, "target_per_dataset":per_dataset,
        "excluded_task_ids":[row["task_id"] for row in exclude],
        "pilot_per_dataset":8, "selection": "uniform seeded without replacement; MATH subject/level round robin",
        "sources": {name: {"path":str(path), "sha256":sha256(path), "rows":len(datasets[name])} for name,path in files.items()},
        "sample_sha256":sha256(output/"samples.jsonl"),
        "exclude_from_future_independent_test_if_used_for_training":True}, indent=2))
    return samples


if __name__ == "__main__":
    prepare()
