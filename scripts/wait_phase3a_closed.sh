#!/usr/bin/env bash
set -u

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
log_file="$project_dir/outputs/phase3a/logs/closed_watcher.log"
minimum_free_mib=12288
maximum_utilization=10
poll_seconds=300

mkdir -p "$(dirname "$log_file")"
exec >>"$log_file" 2>&1
echo "watcher_started $(date -Is) gpus=all minimum_free_mib=$minimum_free_mib maximum_utilization=$maximum_utilization"

while true; do
    if pgrep -f '^([^[:space:]]*/)?python([0-9.]*)? -m dari[.]phase3a closed([[:space:]]|$)' >/dev/null; then
        echo "existing_closed_process $(date -Is), waiting"
        sleep "$poll_seconds"
        continue
    fi

    eligible=()
    if ! gpu_rows=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits); then
        sleep "$poll_seconds"
        continue
    fi
    while IFS=, read -r index free utilization; do
        index="$(echo "$index" | xargs)"
        free="$(echo "$free" | xargs)"
        utilization="$(echo "$utilization" | xargs)"
        echo "gpu_check $(date -Is) gpu=$index free_mib=$free utilization=$utilization"
        if [[ ! "$index" =~ ^[0-9]+$ || ! "$free" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]]; then
            continue
        fi
        if [ "$free" -ge "$minimum_free_mib" ] && [ "$utilization" -le "$maximum_utilization" ]; then
            eligible+=("$index")
        fi
    done <<< "$gpu_rows"

    if [ "${#eligible[@]}" -ge 4 ]; then
        selected_gpus=$(IFS=,; echo "${eligible[*]:0:4}")
        echo "resources_ready $(date -Is), starting closed-loop evaluation on GPUs $selected_gpus"
        cd "$project_dir" || exit 1
        export CUDA_VISIBLE_DEVICES="$selected_gpus"
        export PYTHONUNBUFFERED=1
        .venv/bin/python -m dari.phase3a closed --stop 12
        status=$?
        if [ "$status" -eq 0 ] && .venv/bin/python - <<'VERIFY' && .venv/bin/python -m dari.phase3a_report
import json
from pathlib import Path

root = Path("outputs/phase3a")
samples = [json.loads(line) for line in (root / "splits/eval/samples.jsonl").read_text().splitlines()]
selected = sorted(samples, key=lambda row: (row["selection_rank"], row["dataset"]))[:12]
assert len(selected) == 12
for sample in selected:
    task_id = sample["task_id"]
    for mode in ("anchor", "repair", "help", "bootstrap", "learned", "extended"):
        path = root / "closed" / task_id.replace(":", "_") / f"{mode}.json"
        row = json.loads(path.read_text())
        assert row["task_id"] == task_id and row["mode"] == mode and "reward" in row["score"], path
VERIFY
        then
            echo "closed_completed $(date -Is) verified_results=72"
            exit 0
        fi
        echo "attempt_incomplete $(date -Is) closed_status=$status, retrying after ${poll_seconds}s"
    fi
    sleep "$poll_seconds"
done
