# PageHeat Report Outline

## 1. Introduction

- Motivation: production serving allocates KV in pages, while many eviction methods score tokens.
- Claim: page-native eviction is a better fit for paged serving stacks, especially on agentic traces.
- Contributions:
  - Page-granular learned eviction policy.
  - Agent-trace benchmark for KV retention.
  - Accuracy/throughput evaluation versus Full, StreamingLLM, SnapKV, and optional SAGE-KV.

## 2. Related Work

- PagedAttention / vLLM
- StreamingLLM
- SnapKV
- H2O
- SAGE-KV
- L2-norm-based KV scoring
- FastKV

## 3. Method

### 3.1 Problem setup
- Retention budget.
- Page size.
- Prefill vs decode.

### 3.2 Features
- Mean attention per page.
- Head variance.
- Position features.
- Key-norm feature.
- Page age.

### 3.3 Predictor
- Shared MLP.
- Parameter count.
- Binary target definition.

### 3.4 Eviction policy
- Hard-pinned sink page.
- Hard-pinned recent pages.
- Budget-constrained page selection.

## 4. Experiments

### 4.1 Benchmarks
- LongBench.
- Agent traces.
- Needle-in-a-Haystack.

### 4.2 Metrics
- Accuracy vs retention.
- Decode tokens/s.
- TTFT.
- Peak memory.

### 4.3 Main results
- Headline accuracy-retention plots.
- Throughput table.

### 4.4 Ablations
- Predictor size.
- Logistic vs MLP.
- Page size.
- Sink pinning.

## 5. Failure Analysis and Limitations

- Cases where PageHeat underperforms SnapKV.
- Prompt-length buckets.
- Task categories.
- Current implementation limits: HF path first, vLLM port as stretch.

## 6. Conclusion

- Summarize where PageHeat helps.
- Be explicit that gains are workload-sensitive rather than universal.
