from __future__ import annotations

import argparse
import json
import traceback

from pageheat_app.settings import DEFAULT_PROMPT, ModelConfig


def run_smoke_test(attn_implementation: str) -> dict:
    from pageheat_app.cache_policies import build_cache_policy
    from pageheat_app.modeling import greedy_generate, load_model_and_tokenizer, model_config_dict

    model_config = ModelConfig(attn_implementation=attn_implementation, max_new_tokens=50)
    model, tokenizer = load_model_and_tokenizer(model_config)
    policy = build_cache_policy(
        policy="full",
        retention_rate=1.0,
        sink_tokens=4,
        recent_window=512,
        observation_window=32,
    )
    result = greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=DEFAULT_PROMPT,
        cache_policy=policy,
        max_new_tokens=model_config.max_new_tokens,
    )
    result["model"] = model_config_dict(model_config)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Qwen smoke test.")
    parser.add_argument(
        "--attn",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa"],
        help="Attention backend to request from transformers.",
    )
    args = parser.parse_args(argv)

    try:
        result = run_smoke_test(args.attn)
    except Exception as exc:
        if args.attn == "flash_attention_2":
            fallback = {
                "requested_attn": args.attn,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "fallback": "Retry with --attn sdpa",
            }
            print(json.dumps(fallback, indent=2))
            return 1
        raise

    print(json.dumps(result, indent=2))
    return 0
