from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pageheat_app.utils import read_json, write_json


PredictionRow = dict[str, Any]


def _load_reports(results_dir: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        payload = read_json(path)
        if "summary" in payload and "predictions" in payload:
            reports.append(payload)
    return reports


def _find_report(reports: list[dict[str, Any]], dataset: str, policy: str, retention_rate: float) -> dict[str, Any] | None:
    for report in reports:
        summary = report["summary"]
        cache_policy = summary["cache_policy"]
        if summary["dataset"] != dataset:
            continue
        if cache_policy["policy"] != policy:
            continue
        if float(cache_policy["retention_rate"]) != float(retention_rate):
            continue
        return report
    return None


def _prompt_bucket(cache_trace: list[list[int]] | None) -> str:
    if not cache_trace:
        return "unknown"
    initial = cache_trace[0][0] if cache_trace[0] else 0
    if initial < 4096:
        return "<4k"
    if initial < 16384:
        return "4k-16k"
    if initial < 32768:
        return "16k-32k"
    return "32k+"


def analyze_failures(results_dir: Path, dataset: str, retention_rate: float = 0.2) -> dict[str, Any]:
    reports = _load_reports(results_dir)
    pageheat = _find_report(reports, dataset, "pageheat", retention_rate)
    snapkv = _find_report(reports, dataset, "snapkv", retention_rate)
    full = _find_report(reports, dataset, "full", 1.0)
    if pageheat is None or snapkv is None:
        raise FileNotFoundError(
            f"Need pageheat and snapkv reports for dataset={dataset} retention_rate={retention_rate}."
        )

    pageheat_predictions = {row["sample_index"]: row for row in pageheat["predictions"]}
    snapkv_predictions = {row["sample_index"]: row for row in snapkv["predictions"]}
    full_predictions = {row["sample_index"]: row for row in full["predictions"]} if full else {}

    failures = []
    task_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    task_gap_totals: defaultdict[str, float] = defaultdict(float)

    shared_indices = sorted(set(pageheat_predictions) & set(snapkv_predictions))
    for sample_index in shared_indices:
        pageheat_row = pageheat_predictions[sample_index]
        snapkv_row = snapkv_predictions[sample_index]
        gap = float(pageheat_row["score"]) - float(snapkv_row["score"])
        if gap >= 0:
            continue
        task = pageheat_row.get("task", "unknown")
        bucket = _prompt_bucket(pageheat_row.get("cache_trace"))
        task_counter[task] += 1
        bucket_counter[bucket] += 1
        task_gap_totals[task] += gap
        failures.append(
            {
                "sample_index": sample_index,
                "task": task,
                "pageheat_score": pageheat_row["score"],
                "snapkv_score": snapkv_row["score"],
                "full_score": full_predictions.get(sample_index, {}).get("score"),
                "score_gap": gap,
                "prompt_bucket": bucket,
                "pageheat_prediction": pageheat_row.get("prediction"),
                "snapkv_prediction": snapkv_row.get("prediction"),
                "target": pageheat_row.get("target"),
            }
        )

    failures.sort(key=lambda row: row["score_gap"])
    task_summary = [
        {
            "task": task,
            "count": count,
            "mean_score_gap": task_gap_totals[task] / count,
        }
        for task, count in task_counter.most_common()
    ]

    return {
        "results_dir": str(results_dir),
        "dataset": dataset,
        "retention_rate": retention_rate,
        "num_failures": len(failures),
        "task_summary": task_summary,
        "prompt_bucket_summary": dict(bucket_counter),
        "worst_cases": failures[:25],
    }


def print_markdown(summary: dict[str, Any]) -> None:
    print(f"# Failure Analysis: {summary['dataset']}")
    print()
    print(f"Retention rate: {summary['retention_rate']}")
    print(f"PageHeat worse than SnapKV on {summary['num_failures']} samples")
    print()
    print("## Task Breakdown")
    print()
    print("| task | count | mean gap |")
    print("|---|---:|---:|")
    for row in summary["task_summary"]:
        print(f"| {row['task']} | {row['count']} | {row['mean_score_gap']:.4f} |")
    print()
    print("## Worst Cases")
    print()
    print("| idx | task | pageheat | snapkv | full | gap | bucket |")
    print("|---:|---|---:|---:|---:|---:|---|")
    for row in summary["worst_cases"]:
        full_score = "-" if row["full_score"] is None else f"{row['full_score']:.4f}"
        print(
            f"| {row['sample_index']} | {row['task']} | {row['pageheat_score']:.4f} | {row['snapkv_score']:.4f} | {full_score} | {row['score_gap']:.4f} | {row['prompt_bucket']} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze where PageHeat underperforms SnapKV.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="agent_traces")
    parser.add_argument("--retention-rate", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args(argv)

    summary = analyze_failures(args.results_dir, args.dataset, args.retention_rate)
    if args.output is not None:
        write_json(args.output, summary)
    if args.format == "markdown":
        print_markdown(summary)
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
