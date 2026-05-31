from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from pageheat_app.eval_harness import EvalArgs, run_and_save_eval
from pageheat_app.modeling import load_model_and_tokenizer
from pageheat_app.settings import ModelConfig, RESULTS_DIR
from pageheat_app.utils import ensure_dir, now_ts, write_json


def build_sweep_args(
    config: dict,
    model_id: str,
    attn_implementation: str,
    results_dir: Path,
    sink_tokens: int,
    recent_window: int,
    observation_window: int,
    page_size: int = 16,
    pin_sink_page: bool = True,
    pin_recent_pages: int = 2,
    pageheat_attention_threshold: float = 0.01,
    pageheat_predictor_path: str | None = None,
) -> list[EvalArgs]:
    eval_args_list: list[EvalArgs] = []
    if "experiments" in config:
        experiments = config["experiments"]
    else:
        experiments = []
        for policy in config["cache_policies"]:
            rates = [1.0] if policy == "full" else config["retention_rates"]
            for retention_rate in rates:
                experiments.append(
                    {
                        "name": f"{policy}_r{retention_rate}",
                        "cache_policy": policy,
                        "retention_rate": retention_rate,
                    }
                )

    for dataset in config["datasets"]:
        for experiment in experiments:
            policy = experiment["cache_policy"]
            retention_rate = experiment.get("retention_rate", 1.0)
            policy_attn = experiment.get("attn_implementation")
            if policy_attn is None:
                # Attention-observing policies need eager on the current stack.
                policy_attn = "eager" if policy in {"snapkv", "pageheat"} else attn_implementation
            eval_args_list.append(
                EvalArgs(
                    dataset=dataset,
                    experiment_name=experiment.get("name"),
                    model_id=model_id,
                    attn_implementation=policy_attn,
                    cache_policy=policy,
                    retention_rate=retention_rate,
                    sink_tokens=experiment.get("sink_tokens", sink_tokens),
                    recent_window=experiment.get("recent_window", recent_window),
                    observation_window=experiment.get("observation_window", observation_window),
                    page_size=experiment.get("page_size", page_size),
                    pin_sink_page=experiment.get("pin_sink_page", pin_sink_page),
                    pin_recent_pages=experiment.get("pin_recent_pages", pin_recent_pages),
                    pageheat_attention_threshold=experiment.get("pageheat_attention_threshold", pageheat_attention_threshold),
                    pageheat_predictor_path=experiment.get("pageheat_predictor_path", pageheat_predictor_path),
                    max_new_tokens=experiment.get("max_new_tokens", config.get("max_new_tokens", 128)),
                    max_prompt_tokens=experiment.get("max_prompt_tokens", config.get("max_prompt_tokens")),
                    max_samples=experiment.get("max_samples", config.get("max_samples")),
                    results_dir=results_dir,
                )
            )
    return eval_args_list


def _release_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_sweep(
    config_path: Path,
    model_id: str,
    attn_implementation: str,
    results_dir: Path,
    sink_tokens: int,
    recent_window: int,
    observation_window: int,
    page_size: int = 16,
    pin_sink_page: bool = True,
    pin_recent_pages: int = 2,
    pageheat_attention_threshold: float = 0.01,
    pageheat_predictor_path: str | None = None,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = ensure_dir(results_dir) / f"sweep_{now_ts()}"
    ensure_dir(run_dir)
    summaries = []
    reports: list[dict] = []

    eval_args_list = build_sweep_args(
        config=config,
        model_id=model_id,
        attn_implementation=attn_implementation,
        results_dir=run_dir,
        sink_tokens=sink_tokens,
        recent_window=recent_window,
        observation_window=observation_window,
        page_size=page_size,
        pin_sink_page=pin_sink_page,
        pin_recent_pages=pin_recent_pages,
        pageheat_attention_threshold=pageheat_attention_threshold,
        pageheat_predictor_path=pageheat_predictor_path,
    )

    grouped: dict[tuple[str, str], list[EvalArgs]] = {}
    for eval_args in eval_args_list:
        key = (eval_args.model_id, eval_args.attn_implementation)
        grouped.setdefault(key, []).append(eval_args)

    for (group_model_id, group_attn), group_args in grouped.items():
        model_config = ModelConfig(
            model_id=group_model_id,
            attn_implementation=group_attn,
        )
        model, tokenizer = load_model_and_tokenizer(model_config)
        try:
            for eval_args in group_args:
                report, output_path = run_and_save_eval(eval_args, model=model, tokenizer=tokenizer)
                summaries.append({**report["summary"], "report_path": str(output_path)})
                reports.append({"report_path": str(output_path), "report": report})
        finally:
            _release_model(model)
            del tokenizer

    payload = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "runs": summaries,
        "reports": reports,
        "eval_args": [asdict(item) for item in eval_args_list],
    }
    write_json(run_dir / "summary.json", {k: v for k, v in payload.items() if k != "reports"})
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep Week 1 baseline runs from config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--observation-window", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pin-sink-page", action="store_true", default=True)
    parser.add_argument("--no-pin-sink-page", action="store_false", dest="pin_sink_page")
    parser.add_argument("--pin-recent-pages", type=int, default=2)
    parser.add_argument("--pageheat-attention-threshold", type=float, default=0.01)
    parser.add_argument("--pageheat-predictor-path", default=None)
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
    print(json.dumps({"run_dir": payload["run_dir"], "num_runs": len(payload["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
