# PageHeat

Week 1 results so far: [docs/week1_results.md](docs/week1_results.md) (agent_traces; LongBench-v2 in progress).

Week 1 implementation through the end-of-week SnapKV decision check for:

- Modal smoke test on `Qwen/Qwen2.5-7B-Instruct`
- Full-cache, StreamingLLM, and SnapKV-style baselines
- LongBench-v2 subset evaluation harness
- Agent-trace dataset construction
- Plotting and reporting

This repo is intentionally trimmed to the pieces needed to execute the week plan. The current implementation uses a manual greedy decode runner with pluggable KV-pruning policies and Modal-backed remote evaluation so the expensive baseline runs happen on the A100 path instead of on your MacBook Air.

## Status

- The checked-in results are the Week 1 baselines in [docs/week1_results.md](/Users/HP/PageHeat/docs/week1_results.md).
- The stronger Week 2/PageHeat claim (`96%` of full-cache accuracy at `20%` KV retention, `+8` points over SnapKV, `1.8x` decode throughput at `32K`) is **not** reproduced by checked-in artifacts in this repo yet.
- The codebase now includes the end-to-end PageHeat design path: page-level feature collection, predictor training, page-granular eviction, and sweep configs for Week 2 evaluation. You still need to run the remote data collection, training, and sweeps to produce exact numbers.

## Layout

- `pageheat_app/` core package
- `configs/` benchmark and dataset source configs
- `docs/week1_design.md` prefill/decode hook notes
- `docs/week2_design.md` predictor and page-level eviction notes
- `docs/report_template.md` 4-page writeup skeleton
- `docs/vllm_stretch.md` bounded plan for the vLLM port
- `modal_app.py` Modal entrypoint
- `smoke_test.py` local smoke-test entrypoint
- `tests/` lightweight unit tests for cache policies and metrics

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is for local orchestration on your current machine.
The Modal GPU runtime is defined directly in [pageheat_app/modal_app.py](/Users/HP/PageHeat/pageheat_app/modal_app.py).

On a MacBook Air with Python 3.13, do not expect local `torch==2.4.1` + `flash-attn` to install. Use Modal for the actual Qwen smoke test and benchmark runs.
On Modal, the image now uses `nvidia/cuda:12.4.1-devel-ubuntu22.04` to reduce `flash-attn` build failures that are common on `debian_slim`.

For local evaluation and dataset work, `requirements.txt` includes the CPU-safe `transformers` stack. You still should not expect local `flash-attn`.

## Day 1 smoke test

Local CLI wiring check:

```bash
python smoke_test.py --help
```

Modal:

```bash
modal run modal_app.py
```

If `flash-attn` fails, rerun with `--attn sdpa` and fix the image build separately.

## Build the agent dataset

```bash
python -m pageheat_app.build_agent_dataset \
  --config configs/agent_sources.yaml \
  --output-dir artifacts/data/agent_traces
```

The default agent dataset config now uses three public sources that are directly fetchable:

- `sierra-research/tau-bench` historical airline trajectories from GitHub
- `sierra-research/tau-bench` historical retail trajectories from GitHub
- BFCL V3 multi-turn `miss_param` data from Hugging Face raw JSONL

Quality checks:

```bash
python -m pageheat_app.dataset_qc \
  --dataset-dir artifacts/data/agent_traces
```

The QC output now includes source counts, task counts, duplicate prompts, and an explicit contamination note. It does not claim to prove absence of pretraining contamination, because that cannot be determined reliably from public benchmark artifacts alone.

## Run baselines

Benchmark choice:

- The repo now targets the newer LongBench-v2 benchmark.
- LongBench-v2 uses a multiple-choice schema rather than free-form generation targets.
- The current default subsets are `government_single`, `government_multi`, and `dialogue_history_qa`.
- These subset names come from the public `recursal/longbench-v2` mirror of the December 2024 LongBench-v2 release, while the official dataset card documents the shared LongBench-v2 schema and loading path.

Full cache:

```bash
python -m pageheat_app.remote_eval eval \
  --dataset longbench_v2_government_single \
  --cache-policy full \
  --attn-implementation sdpa \
  --max-samples 100
```

StreamingLLM:

```bash
python -m pageheat_app.remote_eval eval \
  --dataset longbench_v2_government_multi \
  --cache-policy streamingllm \
  --attn-implementation sdpa \
  --retention-rate 0.2
```

SnapKV-style:

```bash
python -m pageheat_app.remote_eval eval \
  --dataset agent_traces \
  --cache-policy snapkv \
  --attn-implementation sdpa \
  --retention-rate 0.1
```

Sweep the LongBench-v2-only matrix:

```bash
python -m pageheat_app.remote_eval sweep \
  --config configs/longbench_tasks.yaml \
  --attn-implementation sdpa
```

Sweep the full Week 1 matrix, including `agent_traces` for the final decision:

```bash
python -m pageheat_app.remote_eval sweep \
  --config configs/week1_matrix.yaml \
  --attn-implementation sdpa
```

End-of-week decision check:

```bash
python -m pageheat_app.decision_check \
  --results-dir artifacts/results/<your_run_dir>
```

All-in-one Week 1 path:

```bash
python -m pageheat_app.week1_runbook
```

## Plot results

```bash
python -m pageheat_app.plot_results \
  --results-dir artifacts/results/<your_run_dir> \
  --output artifacts/plots/week1_accuracy_vs_retention.png
```

## Train and evaluate PageHeat

Collect page-level training data:

```bash
python -m pageheat_app.remote_collect_pageheat_data \
  --dataset agent_traces \
  --max-samples 100 \
  --page-size 16 \
  --output artifacts/data/pageheat/pageheat_train.pt
```

Train a roughly 500K-parameter predictor:

```bash
python -m pageheat_app.train_pageheat_predictor \
  --dataset artifacts/data/pageheat/pageheat_train.pt \
  --model-type mlp \
  --target-params 500000 \
  --num-hidden-layers 2 \
  --output artifacts/models/pageheat_predictor_500k.pt
```

Evaluate the Week 2 sweep:

```bash
python -m pageheat_app.remote_eval sweep \
  --config configs/week2_matrix.yaml \
  --pageheat-predictor-path artifacts/models/pageheat_predictor_500k.pt
```

## Week 3 local pipeline

Run the Week 3 expansion sweep locally:

```bash
python3 -m pageheat_app.week3_runbook \
  --config configs/week3_eval_expansion.yaml \
  --pageheat-predictor-path artifacts/models/pageheat_predictor_500k.pt
```

This writes:

- `summary.json` inside the generated sweep run directory
- `week3_summary.json` with headline accuracy/throughput deltas
- `week3_failure_analysis.json` with PageHeat-vs-SnapKV regressions
- an accuracy-vs-retention plot under `artifacts/plots/`

Compact failure-debugging slice:

```bash
python3 -m pageheat_app.week3_runbook \
  --config configs/week3_failure_slice.yaml \
  --pageheat-predictor-path artifacts/models/pageheat_predictor_500k.pt
```

Standalone summaries:

```bash
python3 -m pageheat_app.summarize_results \
  --results-dir artifacts/results/<your_run_dir> \
  --format markdown
```

```bash
python3 -m pageheat_app.failure_analysis \
  --results-dir artifacts/results/<your_run_dir> \
  --dataset agent_traces \
  --format markdown
```

Needle benchmark dataset names are built into the harness:

- `needle_32k`
- `needle_64k`

They generate synthetic retrieval prompts locally, so no dataset download is required for those runs.

## Single-command benchmark

```bash
make week3-benchmark PREDICTOR=artifacts/models/pageheat_predictor_500k.pt
```

## What to check

1. `python -m pytest tests` passes.
2. `python smoke_test.py --help` works locally.
3. `modal run modal_app.py` loads and generates 50 tokens remotely.
4. `python -m pageheat_app.remote_eval eval ... --cache-policy full` produces metrics JSON with `accuracy`, `ttft_s`, `decode_tokens_per_s`, and `peak_memory_gb`.
5. `streamingllm` and `snapkv` runs both emit lower retained KV lengths than full cache in the saved trace.
6. `python -m pageheat_app.build_agent_dataset ...` writes a local Hugging Face dataset with split stats.
7. `python -m pageheat_app.decision_check --results-dir ...` emits either `gap_exists` or `pivot_or_refine_benchmark`.

## Important caveat

The `snapkv` policy here is a port-ready baseline scaffold driven by observed attention maps during manual decoding. It is built to let you compare retention policies end-to-end now. If you want strict parity with the official SnapKV repository, replace the scoring logic in `pageheat_app/cache_policies.py` with the exact cluster selection routine after you finish day 2 source reading.
