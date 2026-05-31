from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from pageheat_app.cache_policies import build_cache_policy
from pageheat_app.metrics import exact_match, mcq_exact_match, mean, token_f1
from pageheat_app.modeling import greedy_generate, load_model_and_tokenizer, model_config_dict
from pageheat_app.settings import DATA_DIR, ModelConfig, RESULTS_DIR
from pageheat_app.utils import ensure_dir, now_ts, write_json


@dataclass(slots=True)
class EvalArgs:
    dataset: str
    experiment_name: str | None = None
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    attn_implementation: str = "flash_attention_2"
    cache_policy: str = "full"
    retention_rate: float = 1.0
    sink_tokens: int = 4
    recent_window: int = 512
    observation_window: int = 32
    page_size: int = 16
    pin_sink_page: bool = True
    pin_recent_pages: int = 2
    pageheat_attention_threshold: float = 0.01
    pageheat_predictor_path: str | None = None
    max_new_tokens: int = 128
    max_prompt_tokens: int | None = None
    max_samples: int | None = None
    debug_sample_indices: tuple[int, ...] = ()
    results_dir: Path = RESULTS_DIR


def _make_needle_samples(
    *,
    total_tokens: int,
    max_samples: int | None,
    filler_repeats_per_block: int = 64,
) -> list[dict[str, Any]]:
    needle_templates = [
        ("The launch code is AURORA-17.", "AURORA-17"),
        ("The account alias is SILVER-42.", "SILVER-42"),
        ("The warehouse key is MAPLE-88.", "MAPLE-88"),
        ("The recovery word is HARBOR-31.", "HARBOR-31"),
    ]
    sample_count = max_samples or len(needle_templates)
    samples = []
    for idx in range(sample_count):
        sentence, answer = needle_templates[idx % len(needle_templates)]
        filler_block = " ".join(
            f"Background token stream {idx} segment {block_idx} remains irrelevant."
            for block_idx in range(filler_repeats_per_block)
        )
        half = max(total_tokens // 2, 1)
        prompt = "\n\n".join(
            [
                "You will read a long context and answer with the exact secret string only.",
                "Context:",
                ((filler_block + " ") * half) + sentence + " " + ((filler_block + " ") * half),
                "Question: What is the secret string stated in the context?",
                "Answer with the exact string only.",
            ]
        )
        samples.append(
            {
                "prompt": prompt,
                "target": answer,
                "task": f"needle_{total_tokens}",
                "metric": "exact_match",
                "max_new_tokens": 12,
            }
        )
    return samples


def load_samples(dataset_name: str, max_samples: int | None) -> list[dict[str, Any]]:
    if dataset_name == "agent_traces":
        dataset = load_from_disk(str(DATA_DIR / "agent_traces"))["train"]
        samples = [dict(row) for row in dataset]
    elif dataset_name == "needle_32k":
        samples = _make_needle_samples(total_tokens=32_000, max_samples=max_samples)
    elif dataset_name == "needle_64k":
        samples = _make_needle_samples(total_tokens=64_000, max_samples=max_samples)
    elif dataset_name.startswith("longbench_v2_"):
        subset = dataset_name.removeprefix("longbench_v2_")
        dataset = load_dataset("recursal/longbench-v2", subset, split="train")
        samples = []
        for row in dataset:
            choices = row.get("choices")
            if choices and isinstance(choices, list):
                choice_map = {
                    "A": choices[0] if len(choices) > 0 else "",
                    "B": choices[1] if len(choices) > 1 else "",
                    "C": choices[2] if len(choices) > 2 else "",
                    "D": choices[3] if len(choices) > 3 else "",
                }
            else:
                choice_map = {
                    "A": row.get("choice_A", ""),
                    "B": row.get("choice_B", ""),
                    "C": row.get("choice_C", ""),
                    "D": row.get("choice_D", ""),
                }
            prompt = "\n\n".join(
                [
                    "Read the long context and answer the multiple-choice question.",
                    f"Context:\n{row['context']}",
                    f"Question:\n{row['question']}",
                    "Choices:",
                    f"A. {choice_map['A']}",
                    f"B. {choice_map['B']}",
                    f"C. {choice_map['C']}",
                    f"D. {choice_map['D']}",
                    row.get("answer_prefix", "Reply with the correct answer letter only."),
                ]
            )
            samples.append(
                {
                    "prompt": prompt,
                    "target": row["answer"],
                    "task": subset,
                    "metric": "mcq_exact_match",
                    "max_new_tokens": int(row.get("max_new_tokens", 16)),
                }
            )
    elif dataset_name.startswith("longbench_"):
        task = dataset_name.removeprefix("longbench_")
        dataset = load_dataset("THUDM/LongBench", task, split="test")
        samples = [
            {
                "prompt": row["input"],
                "target": row["answers"][0] if row.get("answers") else "",
                "task": task,
                "metric": "token_f1",
            }
            for row in dataset
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def score_prediction(metric_name: str, prediction: str, target: str) -> float:
    if metric_name == "exact_match":
        return exact_match(prediction, target)
    if metric_name == "mcq_exact_match":
        return mcq_exact_match(prediction, target)
    if metric_name in {"token_f1", "rouge_l_proxy"}:
        return token_f1(prediction, target)
    raise ValueError(f"Unsupported metric: {metric_name}")


def run_eval(args, model=None, tokenizer=None) -> dict[str, Any]:
    model_config = ModelConfig(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    owns_model = model is None or tokenizer is None
    if owns_model:
        model, tokenizer = load_model_and_tokenizer(model_config)
    samples = load_samples(args.dataset, args.max_samples)
    predictions: list[dict[str, Any]] = []
    accuracies: list[float] = []
    ttfts: list[float] = []
    tok_s: list[float] = []
    peaks: list[float] = []

    for idx, sample in enumerate(samples):
        cache_policy = build_cache_policy(
            policy=args.cache_policy,
            retention_rate=args.retention_rate,
            sink_tokens=args.sink_tokens,
            recent_window=args.recent_window,
            observation_window=args.observation_window,
            page_size=args.page_size,
            pin_sink_page=args.pin_sink_page,
            pin_recent_pages=args.pin_recent_pages,
            pageheat_attention_threshold=args.pageheat_attention_threshold,
            pageheat_predictor_path=args.pageheat_predictor_path,
        )
        result = greedy_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=sample["prompt"],
            cache_policy=cache_policy,
            max_new_tokens=sample.get("max_new_tokens", args.max_new_tokens),
            max_prompt_tokens=args.max_prompt_tokens,
            observation_window=args.observation_window,
        )
        score = score_prediction(sample["metric"], result["generated_text"], sample["target"])
        accuracies.append(score)
        ttfts.append(result["ttft_s"])
        tok_s.append(result["decode_tokens_per_s"])
        peaks.append(result["peak_memory_gb"])
        predictions.append(
            {
                "sample_index": idx,
                "task": sample["task"],
                "metric": sample["metric"],
                "target": sample["target"],
                "prediction": result["generated_text"],
                "score": score,
                "cache_trace": result["cache_trace"],
                **({"debug": result["debug"]} if idx in set(args.debug_sample_indices) and result.get("debug") is not None else {}),
            }
        )

    summary = {
        "dataset": args.dataset,
        "experiment_name": args.experiment_name,
        "num_samples": len(samples),
        "accuracy": mean(accuracies),
        "ttft_s": mean(ttfts),
        "decode_tokens_per_s": mean(tok_s),
        "peak_memory_gb": mean(peaks),
        "model": model_config_dict(model_config),
        "cache_policy": {
            "policy": args.cache_policy,
            "retention_rate": args.retention_rate,
            "sink_tokens": args.sink_tokens,
            "recent_window": args.recent_window,
            "observation_window": args.observation_window,
            "page_size": args.page_size,
            "pin_sink_page": args.pin_sink_page,
            "pin_recent_pages": args.pin_recent_pages,
            "pageheat_attention_threshold": args.pageheat_attention_threshold,
            "pageheat_predictor_path": args.pageheat_predictor_path,
        },
        "requested_attn_implementation": args.attn_implementation,
        "effective_attn_implementation": args.attn_implementation,
    }
    return {"summary": summary, "predictions": predictions}


def run_and_save_eval(args: EvalArgs, model=None, tokenizer=None) -> tuple[dict[str, Any], Path]:
    report = run_eval(args, model=model, tokenizer=tokenizer)
    name_bits = [args.dataset, args.cache_policy]
    if args.experiment_name:
        name_bits.append(args.experiment_name)
    output_path = ensure_dir(args.results_dir) / f"{'_'.join(name_bits)}_{now_ts()}.json"
    write_json(output_path, report)
    return report, output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Week 1 baseline evaluation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--cache-policy", default="full", choices=["full", "streamingllm", "snapkv", "pageheat"])
    parser.add_argument("--retention-rate", type=float, default=1.0)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--observation-window", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pin-sink-page", action="store_true", default=True)
    parser.add_argument("--no-pin-sink-page", action="store_false", dest="pin_sink_page")
    parser.add_argument("--pin-recent-pages", type=int, default=2)
    parser.add_argument("--pageheat-attention-threshold", type=float, default=0.01)
    parser.add_argument("--pageheat-predictor-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--debug-sample-indices", default="")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    debug_sample_indices = tuple(
        int(item.strip())
        for item in args.debug_sample_indices.split(",")
        if item.strip()
    )

    report, output_path = run_and_save_eval(
        EvalArgs(
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
            results_dir=args.results_dir,
        )
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"saved_report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
