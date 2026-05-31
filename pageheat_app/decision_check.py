from __future__ import annotations

import argparse
import json
from pathlib import Path

from pageheat_app.utils import read_json, write_json


def _load_summaries(results_dir: Path) -> list[dict]:
    summary_json = results_dir / "summary.json"
    if summary_json.exists():
        payload = read_json(summary_json)
        return payload.get("runs", [])

    summaries: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        report = read_json(path)
        if "summary" in report:
            summaries.append(report["summary"])
    return summaries


def evaluate_snapkv_gap(results_dir: Path, dataset_filter: str = "agent_traces", threshold: float = 0.95) -> dict:
    runs = _load_summaries(results_dir)
    dataset_runs = [run for run in runs if run["dataset"] == dataset_filter]
    full_run = next((run for run in dataset_runs if run["cache_policy"]["policy"] == "full"), None)
    snapkv_runs = sorted(
        [run for run in dataset_runs if run["cache_policy"]["policy"] == "snapkv"],
        key=lambda item: item["cache_policy"]["retention_rate"],
    )

    comparisons = []
    for snapkv in snapkv_runs:
        if full_run is None:
            continue
        full_acc = full_run["accuracy"]
        snap_acc = snapkv["accuracy"]
        ratio = 1.0 if full_acc == 0 else snap_acc / full_acc
        comparisons.append(
            {
                "retention_rate": snapkv["cache_policy"]["retention_rate"],
                "full_accuracy": full_acc,
                "snapkv_accuracy": snap_acc,
                "ratio_to_full": ratio,
                "meets_threshold": ratio >= threshold,
            }
        )

    monotonicity_violations = []
    for prev, curr in zip(comparisons, comparisons[1:]):
        if curr["snapkv_accuracy"] < prev["snapkv_accuracy"]:
            monotonicity_violations.append(
                {
                    "lower_retention_rate": prev["retention_rate"],
                    "lower_retention_accuracy": prev["snapkv_accuracy"],
                    "higher_retention_rate": curr["retention_rate"],
                    "higher_retention_accuracy": curr["snapkv_accuracy"],
                }
            )

    if not comparisons:
        verdict = "insufficient_data"
        summary = "No comparable full-cache and SnapKV runs were found for the requested dataset."
    elif monotonicity_violations:
        verdict = "gap_exists"
        summary = "SnapKV accuracy is non-monotonic across retention settings, so the benchmark still shows unstable behavior."
    elif all(item["meets_threshold"] for item in comparisons):
        verdict = "pivot_or_refine_benchmark"
        summary = "SnapKV stays at or above the threshold relative to full cache across the compared agent-trace runs."
    else:
        verdict = "gap_exists"
        summary = "SnapKV drops below the threshold on at least one agent-trace retention setting, so the benchmark still shows a usable gap."

    return {
        "dataset": dataset_filter,
        "threshold": threshold,
        "comparisons": comparisons,
        "monotonicity_violations": monotonicity_violations,
        "verdict": verdict,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether the Week 1 agent benchmark still shows a meaningful SnapKV gap.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="agent_traces")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args(argv)

    result = evaluate_snapkv_gap(args.results_dir, args.dataset, args.threshold)
    write_json(args.results_dir / "decision_check.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
