# Week 2 Design Notes

## Goal

Implement PageHeat as a page-granular learned KV retention policy that can be trained from full-cache traces and evaluated against Full, StreamingLLM, and SnapKV.

## End-to-end path

1. Run full-cache prefill + short decode horizon and materialize attentions.
2. Aggregate token-level signals into fixed-size pages.
3. Train a learned page-importance predictor on future decode attention targets.
4. During inference, score pages after prefill and decode steps.
5. Keep pinned sink/recent pages and fill the remaining KV budget with the highest-value pages.

## Current implementation

- Feature extraction: `pageheat_app/pageheat.py`
- Training-data collection: `pageheat_app/collect_pageheat_data.py`
- Predictor training: `pageheat_app/train_pageheat_predictor.py`
- Inference-time policy: `pageheat_app/cache_policies.py`
- Evaluation sweeps: `configs/week2_matrix.yaml`, `configs/week2_ablations.yaml`

## Predictor sizing

The trainer can now choose an equal-width MLP close to a target parameter count:

```bash
python -m pageheat_app.train_pageheat_predictor \
  --dataset artifacts/data/pageheat/pageheat_train.pt \
  --target-params 500000 \
  --num-hidden-layers 2
```

The saved checkpoint records `hidden_dims` and `param_count` so evaluation reports can be traced back to the trained architecture.

## Exact-results caveat

The repo does not include checked-in artifacts proving the claimed `96% @ 20% retention`, `+8` over SnapKV, or `1.8x` throughput numbers. Those numbers require rerunning the remote Week 2 pipeline and saving the resulting reports under `artifacts/`.
