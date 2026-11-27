"""Worker entry point for the network-isolated scoring launcher."""
import json
from pathlib import Path
import sys


def humaneval(jobs):
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    ids = {job["task_id"] for job in jobs}
    selected = {key: problems[key] for key in ids}
    truth = get_groundtruth(selected, "dari-selected-v0.1.10-" + "-".join(sorted(key.split("/")[-1] for key in ids)), [])
    for job in jobs:
        key = job["task_id"]
        result = check_correctness("humaneval", 0, selected[key], job["code"], truth[key], fast_check=True)
        yield job["key"], {"base": result["base"][0], "plus": result["plus"][0]}


def livecodebench(jobs):
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem
    ids = {job["task_id"].split("/", 1)[1] for job in jobs}
    with Path("/inputs/dataset").open() as stream:
        rows = {}
        for line in stream:
            row = json.loads(line)
            if str(row["question_id"]) in ids:
                rows[str(row["question_id"])] = CodeGenerationProblem(**row).get_evaluation_sample()
    for job in jobs:
        sample = rows[job["task_id"].split("/", 1)[1]]
        results, metadata = evaluate_generations_by_problem(( [job["code"]], sample, False, 6))
        yield job["key"], {"passed": bool(results[0]) and all(value == True for value in results[0]),
                           "metadata": metadata}


def main():
    dataset = sys.argv[1]
    jobs = json.loads(Path("/inputs/jobs.json").read_text())
    evaluator = humaneval if dataset == "humaneval" else livecodebench
    for key, result in evaluator(jobs):
        print("DARI_SCORE " + json.dumps({"key": key, **result}, default=str), flush=True)


if __name__ == "__main__":
    main()
