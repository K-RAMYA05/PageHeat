from __future__ import annotations

import argparse
import json
from hashlib import sha1
from pathlib import Path

from datasets import load_from_disk

from pageheat_app.settings import DATA_DIR
from pageheat_app.utils import write_json


def run_qc(dataset_dir: Path) -> dict:
    dataset = load_from_disk(str(dataset_dir))["train"]
    prompt_lengths = [len(row["prompt"]) for row in dataset]
    target_lengths = [len(row["target"]) for row in dataset]
    prompt_hashes = [sha1(row["prompt"].encode("utf-8")).hexdigest() for row in dataset]
    unique_prompts = len(set(prompt_hashes))
    duplicates = len(prompt_hashes) - unique_prompts
    source_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    metric_counts: dict[str, int] = {}
    for row in dataset:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
        metric_counts[row["metric"]] = metric_counts.get(row["metric"], 0) + 1

    stats = {
        "num_samples": len(prompt_lengths),
        "min_prompt_chars": min(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_chars": max(prompt_lengths) if prompt_lengths else 0,
        "mean_prompt_chars": (sum(prompt_lengths) / len(prompt_lengths)) if prompt_lengths else 0.0,
        "min_target_chars": min(target_lengths) if target_lengths else 0,
        "max_target_chars": max(target_lengths) if target_lengths else 0,
        "mean_target_chars": (sum(target_lengths) / len(target_lengths)) if target_lengths else 0.0,
        "duplicate_prompt_count": duplicates,
        "source_counts": source_counts,
        "task_counts": task_counts,
        "metric_counts": metric_counts,
        "contamination_check": {
            "status": "manual_review_required",
            "note": "Pretraining contamination cannot be ruled out deterministically from public benchmark files alone. Review source provenance manually before making contamination claims.",
        },
    }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quality checks on the built agent dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR / "agent_traces")
    args = parser.parse_args(argv)

    stats = run_qc(args.dataset_dir)
    write_json(args.dataset_dir / "qc_stats.json", stats)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
