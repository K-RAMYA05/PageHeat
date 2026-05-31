from __future__ import annotations

import argparse
import json
from pathlib import Path

from pageheat_app.build_agent_dataset import build_dataset
from pageheat_app.dataset_qc import run_qc
from pageheat_app.decision_check import evaluate_snapkv_gap
from pageheat_app.remote_eval import run_sweep_remote
from pageheat_app.settings import DATA_DIR, RESULTS_DIR
from pageheat_app.utils import write_json


class SweepArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Week 1 pipeline through the end-of-week SnapKV decision check.")
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent_sources.yaml"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/week1_matrix.yaml"))
    parser.add_argument("--agent-output-dir", type=Path, default=DATA_DIR / "agent_traces")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args(argv)

    dataset_stats = build_dataset(args.agent_config, args.agent_output_dir)
    qc_stats = run_qc(args.agent_output_dir)

    sweep_args = SweepArgs(
        command="sweep",
        config=args.sweep_config,
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        results_dir=args.results_dir,
        sink_tokens=4,
        recent_window=512,
        observation_window=32,
        max_prompt_tokens=None,
    )
    run_dir = run_sweep_remote(sweep_args)
    decision = evaluate_snapkv_gap(run_dir, dataset_filter="agent_traces", threshold=0.95)

    summary = {
        "dataset_stats": dataset_stats,
        "qc_stats": qc_stats,
        "results_dir": str(run_dir),
        "decision": decision,
    }
    write_json(run_dir / "week1_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
