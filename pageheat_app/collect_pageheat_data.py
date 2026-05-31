from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pageheat_app.eval_harness import load_samples
from pageheat_app.modeling import _cache_for_model, _chunked_prefill, _legacy_cache, load_model_and_tokenizer
from pageheat_app.pageheat import build_page_feature_batch, page_ids_from_positions
from pageheat_app.settings import DATA_DIR, ModelConfig
from pageheat_app.utils import ensure_dir, now_ts


def _decode_attention_scores(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    layers = []
    for layer_attn in attentions:
        layer_scores = layer_attn.detach().float().mean(dim=(0, 1, 2))
        layers.append(layer_scores)
    return torch.stack(layers, dim=0).mean(dim=0)


@torch.inference_mode()
def collect_sample(
    model,
    tokenizer,
    sample: dict[str, Any],
    observation_window: int,
    decode_horizon: int,
    page_size: int,
    top_k_pages: int,
    max_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    tokenizer_kwargs = {"return_tensors": "pt"}
    if max_prompt_tokens is not None:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_prompt_tokens
    model_inputs = tokenizer(sample["prompt"], **tokenizer_kwargs)
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    prefill, prefill_attentions = _chunked_prefill(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        observation_window=observation_window,
    )
    past_key_values = _legacy_cache(prefill.past_key_values)
    token_positions = torch.arange(int(input_ids.shape[-1]), device=device, dtype=torch.long)
    feature_batch = build_page_feature_batch(
        attentions=prefill_attentions,
        past_key_values=past_key_values,
        token_positions=token_positions,
        page_size=page_size,
        page_last_attended={},
        decode_step=0,
    )

    prompt_page_ids = page_ids_from_positions(token_positions, page_size)
    unique_prompt_pages = feature_batch.page_ids
    page_attention_future = torch.zeros(unique_prompt_pages.numel(), device=device, dtype=torch.float32)
    generated_ids = [int(prefill.logits[:, -1, :].argmax(dim=-1).item())]
    prompt_length = int(input_ids.shape[-1])
    live_cache = past_key_values

    for step_idx in range(decode_horizon):
        next_input_ids = torch.tensor([[generated_ids[-1]]], device=device)
        decode_attention_mask = torch.ones((1, prompt_length + step_idx + 1), device=device, dtype=attention_mask.dtype)
        decode_position_ids = torch.tensor([[prompt_length + step_idx]], device=device, dtype=torch.long)
        outputs = model(
            input_ids=next_input_ids,
            attention_mask=decode_attention_mask,
            position_ids=decode_position_ids,
            past_key_values=_cache_for_model(model, live_cache),
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        live_cache = _legacy_cache(outputs.past_key_values)
        token_scores = _decode_attention_scores(outputs.attentions)
        prompt_scores = token_scores[:prompt_length]
        for page_idx, page_id in enumerate(unique_prompt_pages.tolist()):
            token_mask = prompt_page_ids == int(page_id)
            if torch.any(token_mask):
                page_attention_future[page_idx] += prompt_scores[token_mask].mean()
        next_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        generated_ids.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    k = min(top_k_pages, int(unique_prompt_pages.numel()))
    targets = torch.zeros_like(page_attention_future)
    if k > 0:
        topk = torch.topk(page_attention_future, k=k).indices
        targets[topk] = 1.0

    return {
        "features": feature_batch.feature_matrix.cpu(),
        "targets": targets.cpu(),
        "page_ids": unique_prompt_pages.cpu(),
        "future_attention": page_attention_future.cpu(),
        "feature_names": feature_batch.feature_names,
        "prompt_tokens": prompt_length,
        "decode_steps": len(generated_ids) - 1,
    }


def collect_dataset(args) -> dict[str, Any]:
    model_config = ModelConfig(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    model, tokenizer = load_model_and_tokenizer(model_config)
    samples = load_samples(args.dataset, args.max_samples)
    collected_features = []
    collected_targets = []
    collected_page_ids = []
    collected_sample_ids = []
    prompt_tokens = []
    decode_steps = []
    feature_names: list[str] | None = None

    try:
        for sample_idx, sample in enumerate(samples):
            row = collect_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                observation_window=args.observation_window,
                decode_horizon=args.decode_horizon,
                page_size=args.page_size,
                top_k_pages=args.top_k_pages,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            if feature_names is None:
                feature_names = row["feature_names"]
            num_pages = int(row["features"].shape[0])
            collected_features.append(row["features"])
            collected_targets.append(row["targets"])
            collected_page_ids.append(row["page_ids"])
            collected_sample_ids.append(torch.full((num_pages,), sample_idx, dtype=torch.long))
            prompt_tokens.append(row["prompt_tokens"])
            decode_steps.append(row["decode_steps"])
    finally:
        del model
        del tokenizer

    features = torch.cat(collected_features, dim=0)
    targets = torch.cat(collected_targets, dim=0)
    page_ids = torch.cat(collected_page_ids, dim=0)
    sample_ids = torch.cat(collected_sample_ids, dim=0)

    payload = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "attn_implementation": args.attn_implementation,
        "page_size": args.page_size,
        "decode_horizon": args.decode_horizon,
        "top_k_pages": args.top_k_pages,
        "max_prompt_tokens": args.max_prompt_tokens,
        "num_samples": len(samples),
        "num_examples": int(features.shape[0]),
        "feature_names": feature_names or [],
        "features": features,
        "targets": targets,
        "page_ids": page_ids,
        "sample_ids": sample_ids,
        "prompt_tokens_per_sample": torch.tensor(prompt_tokens, dtype=torch.long),
        "decode_steps_per_sample": torch.tensor(decode_steps, dtype=torch.long),
    }
    ensure_dir(args.output.parent)
    torch.save(payload, args.output)
    return {
        "output": str(args.output),
        "dataset": args.dataset,
        "num_samples": len(samples),
        "num_examples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect PageHeat predictor training data from full-cache decode traces.")
    parser.add_argument("--dataset", default="agent_traces")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--observation-window", type=int, default=32)
    parser.add_argument("--decode-horizon", type=int, default=64)
    parser.add_argument("--top-k-pages", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=DATA_DIR / f"pageheat/pageheat_train_{now_ts()}.pt")
    args = parser.parse_args(argv)

    result = collect_dataset(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
