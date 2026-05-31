from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import torch

from pageheat_app.cache_policies import CachePolicy, cache_lengths
from pageheat_app.settings import ModelConfig


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def load_model_and_tokenizer(model_config: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Make eval reruns bit-stable so SnapKV signal isn't drowned in cuDNN/SDPA
    # algorithm-pick noise. ~10-20% slower decode but eliminates the run-to-run
    # variance that was masking the actual eviction effect.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=_dtype_from_name(model_config.torch_dtype),
        device_map=model_config.device_map,
        attn_implementation=model_config.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-truncate so MCQ prompts keep the question/choices/answer_prefix at
    # the end, and agent prompts keep the most recent turns. Right-truncation
    # silently dropped the question for LongBench-v2 and reduced full cache to
    # below-random accuracy.
    tokenizer.truncation_side = "left"
    model.eval()
    return model, tokenizer


def _cache_for_model(model, past_key_values):
    if past_key_values is None:
        return None
    if not isinstance(past_key_values, tuple):
        return past_key_values

    from transformers import DynamicCache

    return DynamicCache.from_legacy_cache(past_key_values)


def _legacy_cache(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _chunked_prefill(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    observation_window: int,
):
    """Two-stage prefill so the obs-window attention is materialized on a
    [B, H, obs_window, L] slice instead of full [B, H, L, L]. Required by
    SnapKV; saves ~200x memory at L=6000."""
    seq_len = int(input_ids.shape[-1])
    obs = max(1, min(observation_window, seq_len))

    if seq_len <= obs:
        # Prompt is already short enough that one-shot prefill with
        # output_attentions=True is cheap.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        return outputs, outputs.attentions

    head_ids = input_ids[:, : seq_len - obs]
    head_mask = attention_mask[:, : seq_len - obs]
    head_outputs = model(
        input_ids=head_ids,
        attention_mask=head_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )

    tail_ids = input_ids[:, seq_len - obs :]
    tail_outputs = model(
        input_ids=tail_ids,
        attention_mask=attention_mask,
        past_key_values=_cache_for_model(model, head_outputs.past_key_values),
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )
    return tail_outputs, tail_outputs.attentions


def _summarize_retained_spans(tokenizer, input_ids: torch.Tensor, retained_indices: list[int], span_radius: int = 6, limit: int = 12) -> list[dict[str, object]]:
    if not retained_indices:
        return []
    token_ids = input_ids[0].tolist()
    seq_len = len(token_ids)
    chosen_positions = []
    if len(retained_indices) <= limit:
        chosen_positions = retained_indices
    else:
        step = max(1, len(retained_indices) // limit)
        chosen_positions = retained_indices[::step][:limit]

    spans = []
    for pos in chosen_positions:
        start = max(0, pos - span_radius)
        end = min(seq_len, pos + span_radius + 1)
        snippet = tokenizer.decode(token_ids[start:end], skip_special_tokens=False)
        spans.append(
            {
                "position": pos,
                "start": start,
                "end": end,
                "text": snippet,
            }
        )
    return spans


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    cache_policy: CachePolicy,
    max_new_tokens: int,
    max_prompt_tokens: int | None = None,
    observation_window: int = 32,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer_kwargs = {"return_tensors": "pt"}
    if max_prompt_tokens is not None:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_prompt_tokens
    model_inputs = tokenizer(prompt, **tokenizer_kwargs)
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    needs_attentions = cache_policy.requires_attentions()
    needs_decode_attentions = cache_policy.requires_decode_attentions()

    start = time.perf_counter()
    if needs_attentions:
        prefill, prefill_attentions = _chunked_prefill(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            observation_window=observation_window,
        )
    else:
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        prefill_attentions = None
    ttft = time.perf_counter() - start

    prompt_length = int(input_ids.shape[-1])
    generated_ids = [int(prefill.logits[:, -1, :].argmax(dim=-1).item())]
    past_key_values = cache_policy.prune(_legacy_cache(prefill.past_key_values), prefill_attentions)
    cache_trace = [cache_lengths(past_key_values)]

    step_start = time.perf_counter()
    for step_idx in range(max_new_tokens - 1):
        next_input_ids = torch.tensor([[generated_ids[-1]]], device=device)
        # After SnapKV eviction the cache is shorter than the original prompt,
        # but the cached K's still carry RoPE rotated to their *original*
        # positions. We must (a) pass the new query's position_id as
        # `prompt_length + step_idx` (its original-sequence position) so RoPE
        # rotates the new Q to match, and (b) build attention_mask to match
        # the *current* cache length, not the original prompt length.
        cache_seq_length = past_key_values[0][0].shape[-2] if past_key_values else 0
        decode_attention_mask = torch.ones(
            (1, cache_seq_length + 1), device=device, dtype=attention_mask.dtype
        )
        decode_position_ids = torch.tensor(
            [[prompt_length + step_idx]], device=device, dtype=torch.long
        )
        step = model(
            input_ids=next_input_ids,
            attention_mask=decode_attention_mask,
            position_ids=decode_position_ids,
            past_key_values=_cache_for_model(model, past_key_values),
            use_cache=True,
            output_attentions=needs_decode_attentions,
            return_dict=True,
        )
        next_token = int(step.logits[:, -1, :].argmax(dim=-1).item())
        generated_ids.append(next_token)
        step_attentions = step.attentions if needs_decode_attentions else None
        past_key_values = cache_policy.prune(_legacy_cache(step.past_key_values), step_attentions)
        cache_trace.append(cache_lengths(past_key_values))
        if next_token == tokenizer.eos_token_id:
            break

    decode_elapsed = max(time.perf_counter() - step_start, 1e-6)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    peak_memory = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        torch.cuda.empty_cache()

    debug = None
    snapshot = cache_policy.debug_snapshot()
    if snapshot and "retained_indices_layer0" in snapshot:
        debug = {
            **snapshot,
            "retained_text_spans_layer0": _summarize_retained_spans(
                tokenizer=tokenizer,
                input_ids=input_ids,
                retained_indices=snapshot["retained_indices_layer0"],
            ),
        }

    return {
        "generated_text": generated_text,
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": len(generated_ids),
        "ttft_s": ttft,
        "decode_tokens_per_s": len(generated_ids) / decode_elapsed,
        "peak_memory_gb": peak_memory,
        "cache_trace": cache_trace,
        "cache_policy": cache_policy.describe(),
        "debug": debug,
    }


def model_config_dict(model_config: ModelConfig) -> dict[str, Any]:
    return asdict(model_config)
