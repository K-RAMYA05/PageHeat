from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


PastKeyValues = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class PageFeatureBatch:
    page_ids: torch.Tensor
    token_counts: torch.Tensor
    feature_matrix: torch.Tensor
    feature_names: list[str]
    mean_attention: torch.Tensor
    head_variance: torch.Tensor
    key_norm: torch.Tensor
    age: torch.Tensor
    page_mean_attention: torch.Tensor


def page_ids_from_positions(token_positions: torch.Tensor, page_size: int) -> torch.Tensor:
    if token_positions.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=token_positions.device)
    return torch.div(token_positions, page_size, rounding_mode="floor")


def stable_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.detach().float().flatten().cpu()
    targets = targets.detach().float().flatten().cpu()
    pos = int(targets.sum().item())
    neg = int(targets.numel() - pos)
    if pos == 0 or neg == 0:
        return 0.5
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order]
    tps = torch.cumsum(sorted_targets, dim=0)
    fps = torch.cumsum(1.0 - sorted_targets, dim=0)
    tpr = torch.cat([torch.tensor([0.0]), tps / max(pos, 1)])
    fpr = torch.cat([torch.tensor([0.0]), fps / max(neg, 1)])
    return float(torch.trapz(tpr, fpr).item())


def feature_names_for_shape(num_layers: int, num_heads: int) -> list[str]:
    names = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            names.append(f"mean_attention_l{layer_idx}_h{head_idx}")
    for layer_idx in range(num_layers):
        names.append(f"head_variance_l{layer_idx}")
    names.extend(
        [
            "page_index",
            "distance_from_end",
            "distance_from_bos",
            "key_norm",
            "page_age",
        ]
    )
    return names


def build_page_feature_batch(
    attentions: tuple[torch.Tensor, ...],
    past_key_values: PastKeyValues,
    token_positions: torch.Tensor,
    page_size: int,
    page_last_attended: dict[int, int] | None = None,
    decode_step: int = 0,
) -> PageFeatureBatch:
    if not attentions:
        raise ValueError("PageHeat feature extraction requires attentions.")
    if token_positions.numel() == 0:
        raise ValueError("PageHeat feature extraction requires non-empty token positions.")

    device = token_positions.device
    page_ids = page_ids_from_positions(token_positions, page_size)
    unique_page_ids, inverse = torch.unique(page_ids, sorted=True, return_inverse=True)
    num_pages = int(unique_page_ids.numel())
    num_layers = len(attentions)
    num_heads = int(attentions[0].shape[1])

    token_counts = torch.bincount(inverse, minlength=num_pages).to(device=device, dtype=torch.float32)
    mean_attention = torch.zeros((num_pages, num_layers, num_heads), device=device, dtype=torch.float32)
    head_variance = torch.zeros((num_pages, num_layers), device=device, dtype=torch.float32)

    for layer_idx, layer_attn in enumerate(attentions):
        observed = layer_attn.detach().float().mean(dim=2).squeeze(0)
        for head_idx in range(num_heads):
            sums = torch.zeros(num_pages, device=device, dtype=torch.float32)
            sums.index_add_(0, inverse, observed[head_idx])
            mean_attention[:, layer_idx, head_idx] = sums / token_counts.clamp_min(1.0)
        head_variance[:, layer_idx] = mean_attention[:, layer_idx, :].var(dim=-1, unbiased=False)

    key_norms = torch.zeros((num_pages, num_layers), device=device, dtype=torch.float32)
    for layer_idx, (key, _) in enumerate(past_key_values):
        layer_key = key.detach().float().squeeze(0).mean(dim=0)
        for page_idx in range(num_pages):
            token_mask = inverse == page_idx
            if not torch.any(token_mask):
                continue
            page_key = layer_key[token_mask].mean(dim=0)
            key_norms[page_idx, layer_idx] = page_key.norm(p=2)
    key_norm = key_norms.mean(dim=-1)

    max_page_id = int(unique_page_ids.max().item()) if num_pages else 0
    denom = float(max(max_page_id, 1))
    page_index = unique_page_ids.float()
    distance_from_end = (max_page_id - unique_page_ids).float()
    distance_from_bos = unique_page_ids.float()
    age = torch.zeros(num_pages, device=device, dtype=torch.float32)
    if page_last_attended:
        for idx, page_id in enumerate(unique_page_ids.tolist()):
            last_step = page_last_attended.get(int(page_id), 0)
            age[idx] = float(max(decode_step - last_step, 0))
    age_denom = float(max(decode_step, 1))
    page_mean_attention = mean_attention.mean(dim=(1, 2))

    feature_matrix = torch.cat(
        [
            mean_attention.reshape(num_pages, num_layers * num_heads),
            head_variance.reshape(num_pages, num_layers),
            torch.stack(
                [
                    page_index / denom,
                    distance_from_end / denom,
                    distance_from_bos / denom,
                    key_norm,
                    age / age_denom,
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    return PageFeatureBatch(
        page_ids=unique_page_ids,
        token_counts=token_counts,
        feature_matrix=feature_matrix,
        feature_names=feature_names_for_shape(num_layers, num_heads),
        mean_attention=mean_attention,
        head_variance=head_variance,
        key_norm=key_norm,
        age=age,
        page_mean_attention=page_mean_attention,
    )


class PageHeatPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: int | tuple[int, ...] = 64):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims if int(dim) > 0)
        self.hidden_dim = self.hidden_dims[0] if self.hidden_dims else 0

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class PredictorCheckpoint:
    input_dim: int
    hidden_dim: int
    hidden_dims: tuple[int, ...]
    state_dict: dict[str, torch.Tensor]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    feature_indices: torch.Tensor | None = None
    feature_names: list[str] | None = None
    param_count: int | None = None


def predictor_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def choose_hidden_dims_for_target_params(
    input_dim: int,
    target_params: int,
    num_hidden_layers: int = 2,
    multiple_of: int = 32,
    min_hidden_dim: int = 32,
    max_hidden_dim: int = 4096,
) -> tuple[int, ...]:
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be at least 1.")
    if target_params <= 0:
        raise ValueError("target_params must be positive.")

    best_hidden_dim = min_hidden_dim
    best_gap = None
    for hidden_dim in range(min_hidden_dim, max_hidden_dim + 1, multiple_of):
        probe = PageHeatPredictor(input_dim=input_dim, hidden_dims=(hidden_dim,) * num_hidden_layers)
        gap = abs(predictor_parameter_count(probe) - target_params)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_hidden_dim = hidden_dim
    return (best_hidden_dim,) * num_hidden_layers


class LoadedPageHeatPredictor:
    def __init__(self, checkpoint: PredictorCheckpoint):
        self.model = PageHeatPredictor(checkpoint.input_dim, checkpoint.hidden_dims)
        self.model.load_state_dict(checkpoint.state_dict)
        self.model.eval()
        self.feature_mean = checkpoint.feature_mean.float()
        self.feature_std = checkpoint.feature_std.float().clamp_min(1e-6)
        self.feature_indices = checkpoint.feature_indices.long() if checkpoint.feature_indices is not None else None
        self.feature_names = checkpoint.feature_names or []
        self.hidden_dims = checkpoint.hidden_dims
        self.param_count = checkpoint.param_count if checkpoint.param_count is not None else predictor_parameter_count(self.model)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        chosen = features.float().cpu()
        if self.feature_indices is not None:
            chosen = chosen.index_select(1, self.feature_indices)
        normalized = (chosen - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(normalized)
            probs = torch.sigmoid(logits)
        return probs.to(features.device)


def load_predictor_checkpoint(path: str | Path) -> LoadedPageHeatPredictor:
    payload = torch.load(Path(path), map_location="cpu")
    hidden_dims = payload.get("hidden_dims")
    if hidden_dims is None:
        hidden_dims = (int(payload.get("hidden_dim", 64)),)
    else:
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
    checkpoint = PredictorCheckpoint(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 64)),
        hidden_dims=hidden_dims,
        state_dict=payload["state_dict"],
        feature_mean=payload["feature_mean"],
        feature_std=payload["feature_std"],
        feature_indices=payload.get("feature_indices"),
        feature_names=payload.get("feature_names"),
        param_count=payload.get("param_count"),
    )
    return LoadedPageHeatPredictor(checkpoint)


def save_predictor_checkpoint(
    path: str | Path,
    model: PageHeatPredictor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    feature_indices: torch.Tensor | None = None,
    feature_names: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "hidden_dims": list(model.hidden_dims),
        "state_dict": model.state_dict(),
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "feature_indices": None if feature_indices is None else feature_indices.detach().cpu(),
        "feature_names": feature_names,
        "param_count": predictor_parameter_count(model),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path))
