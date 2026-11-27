"""Evaluate saved candidates with pinned upstream checkers inside Bubblewrap."""
import argparse
import gzip
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path("outputs/phase3b_collaboration/challenging_benchmarks")
DATA = Path("data/challenging")
METHODS = ("standard", "anchor_only", "self_repair", "self_repair_extended", "internal_restart", "help")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("humaneval", "livecodebench"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    dataset = args.dataset
    directory = ROOT / dataset
    sources = sorted((directory / "results").glob("*.json"))[:args.limit]
    jobs = []
    for path in sources:
        row = json.loads(path.read_text())
        for method in METHODS:
            if method in row:
                code = row[method]["answer"] or ""
                fence = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", code, re.S | re.I)
                jobs.append({"key": f"{row['task_id']}:{method}", "task_id": row["task_id"],
                             "code": fence.group(1) if fence else code})
    if not jobs:
        raise ValueError("No candidates")
    with TemporaryDirectory(dir=".work") as temporary:
        inputs = Path(temporary).resolve()
        (inputs / "jobs.json").write_text(json.dumps(jobs))
        if dataset == "humaneval":
            (inputs / "dataset").write_bytes(gzip.decompress((DATA / "humanevalplus-v0.1.10.jsonl.gz").read_bytes()))
        else:
            ids = {job["task_id"].split("/", 1)[1] for job in jobs}
            with (DATA / "livecodebench-test6.jsonl").open() as stream, (inputs / "dataset").open("w") as out:
                for line in stream:
                    if str(json.loads(line)["question_id"]) in ids:
                        out.write(line)
        command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
                   "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                   "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--ro-bind", str(inputs), "/inputs",
                   "--ro-bind", str(Path(".work/evaldeps").resolve()), "/deps",
                   "--ro-bind", str(Path(".work/lcb-pinned").resolve()), "/lcb",
                   "--ro-bind", str(Path("scripts/score_challenging_worker.py").resolve()), "/worker.py",
                   "--chdir", "/tmp", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
                   "--setenv", "PYTHONPATH", "/deps:/lcb", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                   "--setenv", "HUMANEVAL_OVERRIDE_PATH", "/inputs/dataset",
                   "/usr/bin/prlimit", "--as=4294967296", "--cpu=900", "--fsize=536870912", "--nofile=128",
                   "/usr/bin/python3.10", "/worker.py", dataset]
        suffix = "_pilot" if args.limit else ""
        with (directory / f"official_evaluation{suffix}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        rows = [json.loads(line.removeprefix("DARI_SCORE ")) for line in
                (directory / f"official_evaluation{suffix}.log").read_text(errors="replace").splitlines()
                if line.startswith("DARI_SCORE ")]
        if result.returncode or sorted(row["key"] for row in rows) != sorted(job["key"] for job in jobs):
            raise RuntimeError(f"{dataset}: exit {result.returncode}, {len(rows)}/{len(jobs)} scores")
    target = directory / f"official_scores{suffix}.json"
    target.write_text(json.dumps(rows, indent=2) + "\n")
    print(dataset, len(rows))


if __name__ == "__main__":
    main()
