# PageHeat

**Page-granular KV eviction for agentic long-context serving**

PageHeat is a KV-cache eviction pipeline for long-context LLM inference. It is built around a simple premise: production serving systems such as vLLM allocate KV cache in fixed-size pages, but most eviction methods still score and prune at token granularity. That mismatch forces token-level heuristics to be aggregated back into pages at eviction time.

PageHeat makes eviction page-native. Instead of ranking individual tokens, it scores pages directly using cheap aggregate signals such as mean attention, attention variance, position, page age, and head-level entropy. A small learned predictor then decides which pages to retain under a KV budget, while hard-pinning sink and recent pages for stability.

## Core thesis

Existing methods such as H2O, SnapKV, and SAGE-KV are designed around token-level importance. That is a poor fit for paged allocation systems where the real eviction unit is already a page. PageHeat treats the page as the first-class eviction unit and learns page importance directly from agentic workload traces, where cross-turn references, tool-use state, and long-horizon dependencies make naive token heuristics brittle.

## Contributions

- Page-native KV eviction designed to map cleanly onto paged attention systems without token-level bookkeeping
- An agent-workload benchmark for KV eviction built from long multi-turn traces and tool-use conversations
- A tiny learned page-importance predictor, roughly 500K parameters, intended to improve over SnapKV-style heuristics on agent traces while remaining competitive on standard long-context benchmarks

## How it works

PageHeat groups cached tokens into fixed-size pages and maintains page-level features during prefill and decode. At eviction time, the model scores each page and removes the lowest-value pages until the cache fits the retention budget.

The main features used by the predictor are:

- mean attention received by the page
- variance of attention across heads
- page position and distance from the sequence end
- page age since strong recent use
- head-level attention entropy
- optional key-vector norm features

To reduce catastrophic mistakes, PageHeat hard-pins:

- the first sink page
- the most recent pages

This preserves the core StreamingLLM stability trick while allowing the learned scorer to make the rest of the budget tradeoff.

## Project scope

This repository contains the pieces needed to:

- run full-cache and eviction-policy baselines
- build an agent-trace dataset for KV eviction evaluation
- collect page-level training data
- train a page-importance predictor
- evaluate retention policies on long-context tasks
- summarize and plot accuracy, throughput, latency, and memory results

The current implementation is centered on a Hugging Face `transformers` pipeline with a custom cache path and Modal-backed remote execution. A direct vLLM integration is the intended serving-oriented target because vLLM already allocates KV cache in fixed-size pages.

## Stack

- `Qwen2.5-7B-Instruct` as the default working model target
- Hugging Face `transformers`
- `flash-attn` or `sdpa`, depending on environment support
- custom cache logic in `pageheat_app/`
- Modal for remote A100/H100 evaluation
- stretch target: page-native integration with vLLM PagedAttention

## Repo layout

- `pageheat_app/`: core package for cache policies, evaluation, training, and analysis
- `configs/`: benchmark, sweep, and dataset configs
- `docs/`: design notes, result summaries, and reporting templates
- `tests/`: lightweight unit tests
- `smoke_test.py`: local smoke-test entrypoint
- `Makefile`: common benchmark entrypoints

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is intended for local orchestration, testing, and CPU-safe tooling. GPU benchmark runs are expected to use Modal.

## Quick start

Check the CLI wiring:

```bash
python smoke_test.py --help
```

Run unit tests:

```bash
python -m pytest tests
```

Run the Modal smoke test:

```bash
modal run modal_app.py
```

If `flash-attn` is unavailable in the runtime, use `sdpa` first and treat FlashAttention setup as an environment issue, not a blocker for basic validation.

## Main workflows

Build the agent dataset:

```bash
python -m pageheat_app.build_agent_dataset \
  --config configs/agent_sources.yaml \
  --output-dir artifacts/data/agent_traces
```

Run dataset QC:

```bash
python -m pageheat_app.dataset_qc \
  --dataset-dir artifacts/data/agent_traces
```

Run a single evaluation:

```bash
python -m pageheat_app.remote_eval eval \
  --dataset longbench_v2_government_single \
  --cache-policy full \
  --attn-implementation sdpa \
  --max-samples 100
```

Run a sweep:

```bash
python -m pageheat_app.remote_eval sweep \
  --config configs/week2_matrix.yaml \
  --attn-implementation sdpa
```

Collect page-level training data:

```bash
python -m pageheat_app.remote_collect_pageheat_data \
  --dataset agent_traces \
  --max-samples 100 \
  --page-size 16 \
  --output artifacts/data/pageheat/pageheat_train.pt
```

Train the predictor:

```bash
python -m pageheat_app.train_pageheat_predictor \
  --dataset artifacts/data/pageheat/pageheat_train.pt \
  --model-type mlp \
  --target-params 500000 \
  --num-hidden-layers 2 \
  --output artifacts/models/pageheat_predictor_500k.pt
```

Evaluate with the trained predictor:

```bash
python -m pageheat_app.remote_eval sweep \
  --config configs/week2_matrix.yaml \
  --pageheat-predictor-path artifacts/models/pageheat_predictor_500k.pt
```

Plot results:

```bash
python -m pageheat_app.plot_results \
  --results-dir artifacts/results/<your_run_dir> \
  --output artifacts/plots/pageheat_accuracy_vs_retention.png
```

## Evaluation focus

The intended evaluation dimensions are:

- accuracy under KV retention budgets such as 10%, 20%, and 50%
- decode throughput
- time to first token
- peak GPU memory
- robustness on both standard long-context tasks and agentic multi-turn workloads

The main claim this repo is structured to test is not that PageHeat must dominate every generic benchmark. The stronger claim is that page-native eviction is a better fit for serving systems and that agentic workloads expose weaknesses in existing token-native heuristics.

## Current positioning

PageHeat is designed as a practical research prototype:

- simple enough to train and iterate on quickly
- aligned with real paged serving systems
- targeted at long-context agent workloads rather than only static QA benchmarks

This makes it relevant to teams working on production inference stacks where KV-cache efficiency, throughput, and memory pressure matter directly.

## Docs

- [Week 1 results](docs/week1_results.md)
- [Week 1 design](docs/week1_design.md)
- [Week 2 design](docs/week2_design.md)
- [vLLM stretch plan](docs/vllm_stretch.md)
- [Report template](docs/report_template.md)

## Limitations

- The current codebase is research-oriented and not yet a polished production serving integration
- Exact parity with external baselines such as official SnapKV implementations should not be assumed without side-by-side verification
- The strongest value of the approach depends on the agent-workload setting being measured honestly alongside standard benchmarks
