from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pageheat_app.eval_harness import EvalArgs
from pageheat_app.modal_app import app, remote_eval
from pageheat_app.settings import RESULTS_DIR
from pageheat_app.utils import ensure_dir, json_default, write_json


def run_smoke(args) -> dict:
    eval_args = EvalArgs(
        dataset=args.dataset,
        experiment_name=args.experiment_name,
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        cache_policy="pageheat",
        retention_rate=args.retention_rate,
        sink_tokens=args.sink_tokens,
        recent_window=args.recent_window,
        observation_window=args.observation_window,
        page_size=args.page_size,
        pin_sink_page=args.pin_sink_page,
        pin_recent_pages=args.pin_recent_pages,
        pageheat_attention_threshold=args.pageheat_attention_threshold,
        pageheat_predictor_path=args.pageheat_predictor_path,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        max_samples=args.max_samples,
        debug_sample_indices=(0,),
        results_dir=Path("/tmp/pageheat_results"),
    )
    agent_config_text = None
    if args.dataset == "agent_traces":
        agent_config_text = Path("configs/agent_sources.yaml").read_text(encoding="utf-8")
    predictor_bytes = None
    if args.pageheat_predictor_path:
        predictor_bytes = Path(args.pageheat_predictor_path).read_bytes()
    payload = json.loads(json.dumps(asdict(eval_args), default=json_default))
    with app.run():
        result = remote_eval.remote(payload, agent_config_text, predictor_bytes)

    report = result["report"]
    first_prediction = report["predictions"][0]
    debug = first_prediction.get("debug") or {}
    if "retained_page_ids" not in debug:
        raise RuntimeError("PageHeat smoke run completed but did not emit retained_page_ids debug output.")

    output_path = ensure_dir(args.results_dir) / Path(result["saved_report"]).name
    write_json(output_path, report)
    return {
        "saved_report": str(output_path),
        "summary": result["summary"],
        "debug_keys": sorted(debug.keys()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal remote PageHeat smoke evaluation.")
    parser.add_argument("--dataset", default="agent_traces")
    parser.add_argument("--experiment-name", default="week2_smoke")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--retention-rate", type=float, default=0.2)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--observation-window", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pin-sink-page", action="store_true", default=True)
    parser.add_argument("--no-pin-sink-page", action="store_false", dest="pin_sink_page")
    parser.add_argument("--pin-recent-pages", type=int, default=2)
    parser.add_argument("--pageheat-attention-threshold", type=float, default=0.01)
    parser.add_argument("--pageheat-predictor-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=4000)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "week2_smoke")
    args = parser.parse_args(argv)

    result = run_smoke(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
