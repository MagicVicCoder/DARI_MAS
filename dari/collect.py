"""Collect theta branch labels or phi receiver preferences without optimization."""
import argparse
import json
from pathlib import Path

from .configuration import load_dataset
from .workflows import collect_phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("controller", "specialist"), required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--allow-specialist-update", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    tasks = load_dataset(args.tasks, require_gold=True)
    print(json.dumps(collect_phase(cfg, tasks, args.output, args.mode, args.controller, args.adapter,
                                   allow_specialist_update=args.allow_specialist_update), indent=2))


if __name__ == "__main__":
    main()
