from __future__ import annotations

import argparse
import json
from pathlib import Path

from pageheat_app.failure_analysis import analyze_failures
from pageheat_app.plot_results import load_points, plot
from pageheat_app.run_baselines import run_sweep
from pageheat_app.settings import PLOTS_DIR, RESULTS_DIR
from pageheat_app.summarize_results import build_summary
from pageheat_app.utils import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Week 3 evaluation, plotting, and failure-analysis pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/week3_eval_expansion.yaml"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--observation-window", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pin-sink-page", action="store_true", default=True)
    parser.add_argument("--no-pin-sink-page", action="store_false", dest="pin_sink_page")
    parser.add_argument("--pin-recent-pages", type=int, default=2)
    parser.add_argument("--pageheat-attention-threshold", type=float, default=0.01)
    parser.add_argument("--pageheat-predictor-path", default=None)
    parser.add_argument("--failure-dataset", default="agent_traces")
    args = parser.parse_args(argv)

    payload = run_sweep(
        config_path=args.config,
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        results_dir=args.results_dir,
        sink_tokens=args.sink_tokens,
        recent_window=args.recent_window,
        observation_window=args.observation_window,
        page_size=args.page_size,
        pin_sink_page=args.pin_sink_page,
        pin_recent_pages=args.pin_recent_pages,
        pageheat_attention_threshold=args.pageheat_attention_threshold,
        pageheat_predictor_path=args.pageheat_predictor_path,
    )

    run_dir = Path(payload["run_dir"])
    summary = build_summary(run_dir)
    failures = analyze_failures(run_dir, dataset=args.failure_dataset, retention_rate=0.2)
    plot_output = args.plots_dir / f"{run_dir.name}_accuracy_vs_retention.png"
    plot(load_points(run_dir), plot_output)

    write_json(run_dir / "week3_summary.json", summary)
    write_json(run_dir / "week3_failure_analysis.json", failures)

    result = {
        "run_dir": str(run_dir),
        "plot": str(plot_output),
        "summary": str(run_dir / "week3_summary.json"),
        "failure_analysis": str(run_dir / "week3_failure_analysis.json"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
