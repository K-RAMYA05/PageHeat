# Week 1 Design Notes

## Prefill vs decode

Prefill runs the full prompt through the model once to create the initial KV cache and the first next-token logits. Decode then feeds one token at a time with `past_key_values` attached. This is where retention methods matter: every decode step either carries the full historical KV tensors forward or prunes them before the next step.

## Hook points

1. Attention score observation happens immediately after each forward pass when `output_attentions=True` is set.
2. KV eviction happens after the forward pass returns `past_key_values` and before the next decode step uses them.
3. For a stricter future HF `Cache` subclass port, the same eviction logic belongs in `update()` or in an explicit `prune()` called right after `update()` inside the subclass.

## Data flow

1. Tokenize prompt.
2. Prefill forward:
   - returns `logits`, `attentions`, `past_key_values`
   - policy observes prefill attentions
   - policy prunes KV tensors to retention budget
3. Decode loop:
   - send previous token and current pruned cache
   - receive updated cache and per-step attentions
   - update policy scores
   - prune again

## Policy behavior

- `full`: no pruning
- `streamingllm`: keep first sink tokens plus trailing recent window
- `snapkv`: keep sink tokens, guaranteed recent window, and highest-scoring historical tokens ranked by observed attention mass from the latest query window

## Why manual decode first

`transformers` cache internals have changed across versions. Manual decode with explicit `past_key_values` pruning gives you an end-to-end benchmark path now, while preserving an obvious migration path to `cache_utils.py` subclasses once you finish source reading and want exact parity with older SnapKV code.
