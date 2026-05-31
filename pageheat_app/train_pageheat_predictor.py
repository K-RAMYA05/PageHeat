from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pageheat_app.pageheat import (
    PageHeatPredictor,
    choose_hidden_dims_for_target_params,
    predictor_parameter_count,
    save_predictor_checkpoint,
    stable_auroc,
)
from pageheat_app.settings import ARTIFACTS_DIR
from pageheat_app.utils import ensure_dir, now_ts


class LogisticPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 0
        self.net = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class SplitDataset:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    feature_names: list[str]
    feature_indices: torch.Tensor


def _parse_hidden_dims(hidden_dims: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in hidden_dims.split(",") if item.strip())
    if not dims:
        raise ValueError("hidden_dims must contain at least one positive integer.")
    if any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dims values must be positive.")
    return dims


def _feature_indices(feature_names: list[str], feature_set: str) -> torch.Tensor:
    if feature_set == "all":
        return torch.arange(len(feature_names), dtype=torch.long)
    if feature_set == "attention_only":
        keep = [idx for idx, name in enumerate(feature_names) if name.startswith("mean_attention") or name.startswith("head_variance")]
        return torch.tensor(keep, dtype=torch.long)
    if feature_set == "no_key_norm":
        keep = [idx for idx, name in enumerate(feature_names) if name != "key_norm"]
        return torch.tensor(keep, dtype=torch.long)
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def load_split(path: Path, holdout_samples: int, feature_set: str) -> SplitDataset:
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    targets = payload["targets"].float()
    sample_ids = payload["sample_ids"].long()
    feature_names = list(payload.get("feature_names", []))
    feature_indices = _feature_indices(feature_names, feature_set)
    features = features.index_select(1, feature_indices)
    selected_feature_names = [feature_names[idx] for idx in feature_indices.tolist()]

    unique_samples = torch.unique(sample_ids, sorted=True)
    holdout = min(holdout_samples, int(unique_samples.numel()) // 2 if unique_samples.numel() > 1 else 1)
    val_ids = unique_samples[-holdout:]
    train_ids = unique_samples[:-holdout] if holdout < unique_samples.numel() else unique_samples[:1]
    train_mask = torch.isin(sample_ids, train_ids)
    val_mask = torch.isin(sample_ids, val_ids)

    return SplitDataset(
        train_x=features[train_mask],
        train_y=targets[train_mask],
        val_x=features[val_mask],
        val_y=targets[val_mask],
        feature_names=selected_feature_names,
        feature_indices=feature_indices,
    )


def build_model(
    model_type: str,
    input_dim: int,
    hidden_dims: tuple[int, ...],
) -> nn.Module:
    if model_type == "mlp":
        return PageHeatPredictor(input_dim=input_dim, hidden_dims=hidden_dims)
    if model_type == "logistic":
        return LogisticPredictor(input_dim=input_dim)
    raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(features)
        probs = torch.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return {
        "loss": float(loss.item()),
        "auroc": stable_auroc(probs, targets),
    }


def train(args) -> dict:
    split = load_split(args.dataset, args.holdout_samples, args.feature_set)
    feature_mean = split.train_x.mean(dim=0)
    feature_std = split.train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = (split.train_x - feature_mean) / feature_std
    val_x = (split.val_x - feature_mean) / feature_std
    train_y = split.train_y
    val_y = split.val_y

    device = torch.device(args.device)
    if args.model_type == "mlp":
        if args.target_params is not None:
            hidden_dims = choose_hidden_dims_for_target_params(
                input_dim=int(train_x.shape[1]),
                target_params=args.target_params,
                num_hidden_layers=args.num_hidden_layers,
            )
        else:
            hidden_dims = _parse_hidden_dims(args.hidden_dims)
    else:
        hidden_dims = ()

    model = build_model(
        args.model_type,
        input_dim=int(train_x.shape[1]),
        hidden_dims=hidden_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_val_auroc = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        train_metrics = evaluate(model, train_x.to(device), train_y.to(device))
        val_metrics = evaluate(model, val_x.to(device), val_y.to(device))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(sum(losses) / max(len(losses), 1)),
            "train_auroc": train_metrics["auroc"],
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
        }
        history.append(record)
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)

    ensure_dir(args.output.parent)
    save_predictor_checkpoint(
        path=args.output,
        model=model.cpu(),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_indices=split.feature_indices,
        feature_names=split.feature_names,
        extra={
            "model_type": args.model_type,
            "feature_set": args.feature_set,
            "history": history,
        },
    )

    logistic_val_auroc = None
    if args.model_type == "mlp":
        baseline_output = args.output.with_name(args.output.stem + "_logistic.pt")
        baseline_args = argparse.Namespace(**{**vars(args), "model_type": "logistic", "output": baseline_output})
        baseline_result = train(baseline_args)
        logistic_val_auroc = baseline_result["best_val_auroc"]

    return {
        "output": str(args.output),
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "input_dim": int(train_x.shape[1]),
        "hidden_dims": list(hidden_dims),
        "param_count": predictor_parameter_count(model),
        "best_val_auroc": best_val_auroc,
        "logistic_val_auroc": logistic_val_auroc,
        "history_tail": history[-5:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the PageHeat predictor on collected page-level data.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / f"models/pageheat_predictor_{now_ts()}.pt")
    parser.add_argument("--model-type", choices=["mlp", "logistic"], default="mlp")
    parser.add_argument("--feature-set", choices=["all", "attention_only", "no_key_norm"], default="all")
    parser.add_argument("--holdout-samples", type=int, default=30)
    parser.add_argument("--hidden-dims", default="64", help="Comma-separated hidden layer widths for the MLP predictor.")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="Used with --target-params to build a width-matched MLP.")
    parser.add_argument("--target-params", type=int, default=None, help="If set, choose an equal-width MLP close to this parameter count.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    result = train(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
