# vLLM Stretch Plan

## Goal

Port PageHeat from the HuggingFace manual-decode path to a vLLM-style block/page eviction path without changing the predictor semantics.

## Mapping

- `PageHeatPolicy` score output maps to block priority.
- vLLM block manager remains the allocator.
- PageHeat decides which blocks are least valuable under a retention budget.

## Minimal scope

1. Reuse the trained predictor checkpoint.
2. Aggregate per-block features compatible with the HF implementation.
3. Add block-priority scoring before eviction.
4. Demonstrate one end-to-end benchmark with throughput and accuracy.

## Bail-out rule

If the port requires deeper scheduler or allocator surgery than a bounded block-priority hook, stop and keep the HF path as the main artifact.
