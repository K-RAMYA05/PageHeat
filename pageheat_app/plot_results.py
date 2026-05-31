from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pageheat_app.utils import read_json


def load_points(results_dir: Path):
    points = []
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        payload = read_json(summary_path)
        for summary in payload.get("runs", []):
            points.append(
                {
                    "dataset": summary["dataset"],
                    "policy": summary["cache_policy"]["policy"],
                    "retention_rate": summary["cache_policy"]["retention_rate"],
                    "accuracy": summary["accuracy"],
                }
            )
        return points

    for path in sorted(results_dir.glob("*.json")):
        report = read_json(path)
        summary = report["summary"]
        points.append(
            {
                "dataset": summary["dataset"],
                "policy": summary["cache_policy"]["policy"],
                "retention_rate": summary["cache_policy"]["retention_rate"],
                "accuracy": summary["accuracy"],
            }
        )
    return points


def plot(points, output: Path) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for point in points:
        grouped.setdefault((point["dataset"], point["policy"]), []).append(point)

    datasets = sorted({dataset for dataset, _ in grouped})
    policy_colors = {
        "full": "#0b6e4f",
        "streamingllm": "#355c7d",
        "snapkv": "#d1495b",
        "pageheat": "#d4a017",
    }
    dataset_linestyles = ["-", "--", "-.", ":"]
    dataset_styles = {dataset: dataset_linestyles[idx % len(dataset_linestyles)] for idx, dataset in enumerate(datasets)}
    dataset_labels = {
        "agent_traces": "agent",
        "longbench_v2_dialogue_history_qa": "dialogue",
        "longbench_v2_government_multi": "gov-multi",
        "longbench_v2_government_single": "gov-single",
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    policy_handles = {}

    for dataset in datasets:
        for policy in ("full", "streamingllm", "snapkv", "pageheat"):
            series = grouped.get((dataset, policy))
            if not series:
                continue
            series = sorted(series, key=lambda item: item["retention_rate"])
            line = ax.plot(
                [item["retention_rate"] for item in series],
                [item["accuracy"] for item in series],
                marker="o",
                linewidth=2,
                color=policy_colors.get(policy),
                linestyle=dataset_styles[dataset],
            )[0]
            policy_handles.setdefault(policy, line)

    ax.set_xlabel("Retention rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs retention")
    ax.grid(True, alpha=0.25)

    policy_legend = ax.legend(
        [policy_handles[name] for name in ("full", "streamingllm", "snapkv", "pageheat") if name in policy_handles],
        [name for name in ("full", "streamingllm", "snapkv", "pageheat") if name in policy_handles],
        title="Policy",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    ax.add_artist(policy_legend)

    dataset_handles = [
        Line2D([0], [0], color="black", linestyle=dataset_styles[dataset], linewidth=2, marker="o")
        for dataset in datasets
    ]
    ax.legend(
        dataset_handles,
        [dataset_labels.get(name, name) for name in datasets],
        title="Dataset",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output, dpi=180)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Week 1 accuracy curves.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    plot(load_points(args.results_dir), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
