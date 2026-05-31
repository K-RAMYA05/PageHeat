from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pageheat_app.eval_harness import EvalArgs
from pageheat_app.modal_app import app, remote_eval, remote_run_sweep
from pageheat_app.settings import RESULTS_DIR
from pageheat_app.utils import ensure_dir, json_default, now_ts, write_json


def _agent_config_text_if_referenced(text: str) -> str | None:
    if "agent_traces" not in text:
        return None
    return Path("configs/agent_sources.yaml").read_text(encoding="utf-8")


def _predictor_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    return Path(path).read_bytes()


def run_single(args) -> Path:
    debug_sample_indices = tuple(
        int(item.strip())
        for item in args.debug_sample_indices.split(",")
        if item.strip()
    )
    eval_args = EvalArgs(
        dataset=args.dataset,
        experiment_name=args.experiment_name,
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        cache_policy=args.cache_policy,
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
        debug_sample_indices=debug_sample_indices,
        results_dir=Path("/tmp/pageheat_results"),
    )
    agent_config_text = None
    if args.dataset == "agent_traces":
        agent_config_text = Path("configs/agent_sources.yaml").read_text(encoding="utf-8")
    predictor_bytes = _predictor_bytes(args.pageheat_predictor_path)
    payload = json.loads(json.dumps(asdict(eval_args), default=json_default))
    with app.run():
        result = remote_eval.remote(payload, agent_config_text, predictor_bytes)
    output_path = ensure_dir(args.results_dir) / Path(result["saved_report"]).name
    write_json(output_path, result["report"])
    print(json.dumps(result["summary"], indent=2))
    print(f"saved_report={output_path}")
    return output_path


def run_sweep_remote(args) -> Path:
    config_text = args.config.read_text(encoding="utf-8")
    agent_config_text = _agent_config_text_if_referenced(config_text)
    predictor_bytes = _predictor_bytes(args.pageheat_predictor_path)

    local_run_dir = ensure_dir(args.results_dir) / f"sweep_{now_ts()}"
    ensure_dir(local_run_dir)

    with app.run():
        payload = remote_run_sweep.remote(
            config_text=config_text,
            agent_config_text=agent_config_text,
            predictor_bytes=predictor_bytes,
            model_id=args.model_id,
            attn_implementation=args.attn_implementation,
            results_dir="/tmp/pageheat_results",
            sink_tokens=args.sink_tokens,
            recent_window=args.recent_window,
            observation_window=args.observation_window,
            page_size=args.page_size,
            pin_sink_page=args.pin_sink_page,
            pin_recent_pages=args.pin_recent_pages,
            pageheat_attention_threshold=args.pageheat_attention_threshold,
            pageheat_predictor_path=args.pageheat_predictor_path,
        )

    summaries: list[dict] = []
    for entry in payload.get("reports", []):
        remote_report_path = Path(entry["report_path"])
        local_path = local_run_dir / remote_report_path.name
        write_json(local_path, entry["report"])
        summary = entry["report"].get("summary", {})
        summaries.append({**summary, "report_path": str(local_path)})

    summary_payload = {
        "run_dir": str(local_run_dir),
        "config_path": str(args.config),
        "runs": summaries,
        "eval_args": payload.get("eval_args", []),
    }
    write_json(local_run_dir / "summary.json", summary_payload)
    print(json.dumps({"run_dir": str(local_run_dir), "num_runs": len(summaries)}, indent=2))
    return local_run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run remote Week 1 evaluations on Modal.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("eval")
    single.add_argument("--dataset", required=True)
    single.add_argument("--experiment-name", default=None)
    single.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    single.add_argument("--attn-implementation", default="flash_attention_2")
    single.add_argument("--cache-policy", default="full", choices=["full", "streamingllm", "snapkv", "pageheat"])
    single.add_argument("--retention-rate", type=float, default=1.0)
    single.add_argument("--sink-tokens", type=int, default=4)
    single.add_argument("--recent-window", type=int, default=512)
    single.add_argument("--observation-window", type=int, default=32)
    single.add_argument("--page-size", type=int, default=16)
    single.add_argument("--pin-sink-page", action="store_true", default=True)
    single.add_argument("--no-pin-sink-page", action="store_false", dest="pin_sink_page")
    single.add_argument("--pin-recent-pages", type=int, default=2)
    single.add_argument("--pageheat-attention-threshold", type=float, default=0.01)
    single.add_argument("--pageheat-predictor-path", default=None)
    single.add_argument("--max-new-tokens", type=int, default=128)
    single.add_argument("--max-prompt-tokens", type=int, default=None)
    single.add_argument("--max-samples", type=int, default=None)
    single.add_argument("--debug-sample-indices", default="")
    single.add_argument("--results-dir", type=Path, default=RESULTS_DIR)

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    sweep.add_argument("--attn-implementation", default="flash_attention_2")
    sweep.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    sweep.add_argument("--sink-tokens", type=int, default=4)
    sweep.add_argument("--recent-window", type=int, default=512)
    sweep.add_argument("--observation-window", type=int, default=32)
    sweep.add_argument("--page-size", type=int, default=16)
    sweep.add_argument("--pin-sink-page", action="store_true", default=True)
    sweep.add_argument("--no-pin-sink-page", action="store_false", dest="pin_sink_page")
    sweep.add_argument("--pin-recent-pages", type=int, default=2)
    sweep.add_argument("--pageheat-attention-threshold", type=float, default=0.01)
    sweep.add_argument("--pageheat-predictor-path", default=None)

    args = parser.parse_args(argv)
    if args.command == "eval":
        run_single(args)
    else:
        run_sweep_remote(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
