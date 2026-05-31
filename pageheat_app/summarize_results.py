from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pageheat_app.utils import read_json, write_json


MetricRow = dict[str, Any]


def _load_runs(results_dir: Path) -> list[MetricRow]:
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        payload = read_json(summary_path)
        return list(payload.get("runs", []))

    runs: list[MetricRow] = []
    for path in sorted(results_dir.glob("*.json")):
        report = read_json(path)
        if "summary" in report:
            runs.append(report["summary"])
    return runs


def _policy_key(run: MetricRow) -> tuple[str, float]:
    policy = run["cache_policy"]["policy"]
    retention = float(run["cache_policy"]["retention_rate"])
    return policy, retention


def build_summary(results_dir: Path) -> dict[str, Any]:
    runs = _load_runs(results_dir)
    grouped: dict[str, dict[tuple[str, float], MetricRow]] = {}
    for run in runs:
        grouped.setdefault(run["dataset"], {})[_policy_key(run)] = run

    datasets = []
    for dataset, entries in sorted(grouped.items()):
        full = entries.get(("full", 1.0))
        dataset_summary = {
            "dataset": dataset,
            "full_accuracy": full["accuracy"] if full else None,
            "rows": [],
        }
        for (policy, retention), row in sorted(entries.items(), key=lambda item: (item[0][0], item[0][1])):
            full_accuracy = full["accuracy"] if full else None
            ratio = None
            if full_accuracy not in (None, 0):
                ratio = row["accuracy"] / full_accuracy
            dataset_summary["rows"].append(
                {
                    "policy": policy,
                    "retention_rate": retention,
                    "accuracy": row["accuracy"],
                    "accuracy_vs_full": ratio,
                    "decode_tokens_per_s": row.get("decode_tokens_per_s"),
                    "throughput_vs_full": (
                        row.get("decode_tokens_per_s", 0.0) / full.get("decode_tokens_per_s", 1.0)
                        if full and full.get("decode_tokens_per_s")
                        else None
                    ),
                    "ttft_s": row.get("ttft_s"),
                    "peak_memory_gb": row.get("peak_memory_gb"),
                }
            )
        datasets.append(dataset_summary)

    headline = []
    for dataset_summary in datasets:
        rows = dataset_summary["rows"]
        snapkv = next((row for row in rows if row["policy"] == "snapkv" and row["retention_rate"] == 0.2), None)
        pageheat = next((row for row in rows if row["policy"] == "pageheat" and row["retention_rate"] == 0.2), None)
        if snapkv and pageheat:
            headline.append(
                {
                    "dataset": dataset_summary["dataset"],
                    "pageheat_accuracy_vs_full": pageheat["accuracy_vs_full"],
                    "pageheat_minus_snapkv": pageheat["accuracy"] - snapkv["accuracy"],
                    "pageheat_throughput_vs_snapkv": (
                        pageheat["decode_tokens_per_s"] / snapkv["decode_tokens_per_s"]
                        if snapkv.get("decode_tokens_per_s")
                        else None
                    ),
                }
            )

    return {
        "results_dir": str(results_dir),
        "num_runs": len(runs),
        "datasets": datasets,
        "headline": headline,
    }


def print_markdown(summary: dict[str, Any]) -> None:
    print("# Results Summary")
    print()
    print(f"Runs: {summary['num_runs']}")
    print()
    for dataset in summary["datasets"]:
        print(f"## {dataset['dataset']}")
        print()
        print("| policy | retention | accuracy | acc/full | tok/s | tok/s full-ratio | peak GB |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for row in dataset["rows"]:
            acc_full = "-" if row["accuracy_vs_full"] is None else f"{row['accuracy_vs_full']:.3f}"
            tok_s = "-" if row["decode_tokens_per_s"] is None else f"{row['decode_tokens_per_s']:.2f}"
            tok_ratio = "-" if row["throughput_vs_full"] is None else f"{row['throughput_vs_full']:.2f}"
            peak = "-" if row["peak_memory_gb"] is None else f"{row['peak_memory_gb']:.2f}"
            print(
                f"| {row['policy']} | {row['retention_rate']:.1f} | {row['accuracy']:.4f} | {acc_full} | {tok_s} | {tok_ratio} | {peak} |"
            )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a sweep into headline accuracy and throughput deltas.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args(argv)

    summary = build_summary(args.results_dir)
    if args.output is not None:
        write_json(args.output, summary)
    if args.format == "markdown":
        print_markdown(summary)
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
