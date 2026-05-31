from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pageheat_app.modal_app import app, remote_collect_pageheat_data
from pageheat_app.settings import DATA_DIR
from pageheat_app.utils import ensure_dir


def run_remote(args) -> dict:
    agent_config_text = None
    if args.dataset == "agent_traces":
        agent_config_text = Path("configs/agent_sources.yaml").read_text(encoding="utf-8")
    payload = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "attn_implementation": args.attn_implementation,
        "page_size": args.page_size,
        "observation_window": args.observation_window,
        "decode_horizon": args.decode_horizon,
        "top_k_pages": args.top_k_pages,
        "max_samples": args.max_samples,
        "max_prompt_tokens": args.max_prompt_tokens,
        "output": "/tmp/pageheat_train.pt",
    }
    with app.run():
        result = remote_collect_pageheat_data.remote(payload, agent_config_text)

    output = ensure_dir(args.output.parent)
    torch.save(result["payload"], args.output)
    return {
        "output": str(args.output),
        **result["summary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect PageHeat training data remotely on Modal.")
    parser.add_argument("--dataset", default="agent_traces")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--observation-window", type=int, default=32)
    parser.add_argument("--decode-horizon", type=int, default=64)
    parser.add_argument("--top-k-pages", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "pageheat/pageheat_train.pt")
    args = parser.parse_args(argv)

    result = run_remote(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
