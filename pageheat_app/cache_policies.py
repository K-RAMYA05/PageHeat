from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from pageheat_app.pageheat import build_page_feature_batch, load_predictor_checkpoint


PastKeyValues = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def _seq_len(layer: tuple[torch.Tensor, torch.Tensor]) -> int:
    return int(layer[0].shape[-2])


def _slice_layer(layer: tuple[torch.Tensor, torch.Tensor], indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key, value = layer
    return key.index_select(-2, indices), value.index_select(-2, indices)


@dataclass
class CachePolicy:
    name: str
    retention_rate: float = 1.0
    sink_tokens: int = 4
    recent_window: int = 512

    def retain_count(self, seq_len: int) -> int:
        return max(self.sink_tokens, int(seq_len * self.retention_rate))

    def prune(
        self,
        past_key_values: PastKeyValues | None,
        attentions: tuple[torch.Tensor, ...] | None = None,
    ) -> PastKeyValues | None:
        return past_key_values

    def describe(self) -> dict[str, float | int | str]:
        return {
            "policy": self.name,
            "retention_rate": self.retention_rate,
            "sink_tokens": self.sink_tokens,
            "recent_window": self.recent_window,
        }

    def requires_attentions(self) -> bool:
        return False

    def requires_decode_attentions(self) -> bool:
        return False

    def debug_snapshot(self) -> dict | None:
        return None


@dataclass
class FullCachePolicy(CachePolicy):
    name: str = "full"


@dataclass
class StreamingLLMPolicy(CachePolicy):
    name: str = "streamingllm"

    def prune(
        self,
        past_key_values: PastKeyValues | None,
        attentions: tuple[torch.Tensor, ...] | None = None,
    ) -> PastKeyValues | None:
        if not past_key_values:
            return past_key_values

        seq_len = _seq_len(past_key_values[0])
        retained = self.retain_count(seq_len)
        recent = min(self.recent_window, max(retained - self.sink_tokens, 0), max(seq_len - self.sink_tokens, 0))
        if seq_len <= self.sink_tokens + recent:
            return past_key_values

        prefix = torch.arange(self.sink_tokens, device=past_key_values[0][0].device)
        suffix = torch.arange(seq_len - recent, seq_len, device=prefix.device)
        indices = torch.cat([prefix, suffix]).unique(sorted=True)
        return tuple(_slice_layer(layer, indices) for layer in past_key_values)

    def debug_snapshot(self) -> dict | None:
        return None


@dataclass
class SnapKVPolicy(CachePolicy):
    name: str = "snapkv"
    observation_window: int = 32
    score_alpha: float = 0.8
    layer_scores: list[torch.Tensor] = field(default_factory=list)
    last_retained_indices: list[torch.Tensor] = field(default_factory=list)

    def requires_attentions(self) -> bool:
        # At retention_rate >= 1.0 no eviction happens, so there's nothing to
        # score. Skipping the chunked prefill in that case keeps SnapKV's
        # numerics bit-identical to the full-cache path and eliminates the
        # ~2 F1 drift caused by SDPA-head + eager-tail kernel mismatch.
        return self.retention_rate < 1.0

    def _ensure_scores(self, past_key_values: PastKeyValues) -> None:
        if self.layer_scores:
            return
        for layer in past_key_values:
            device = layer[0].device
            self.layer_scores.append(torch.zeros(_seq_len(layer), device=device, dtype=torch.float32))

    def _resize_scores(self, past_key_values: PastKeyValues) -> None:
        for idx, layer in enumerate(past_key_values):
            target_len = _seq_len(layer)
            scores = self.layer_scores[idx]
            if scores.numel() == target_len:
                continue
            if scores.numel() < target_len:
                extra = torch.zeros(target_len - scores.numel(), device=scores.device, dtype=scores.dtype)
                self.layer_scores[idx] = torch.cat([scores, extra], dim=0)
            else:
                self.layer_scores[idx] = scores[-target_len:]

    def _observe_attentions(
        self,
        attentions: tuple[torch.Tensor, ...],
        past_key_values: PastKeyValues,
    ) -> None:
        # attentions[i] has shape [B, H, obs_window, L_total] — captured from
        # the chunked prefill's second pass (last `observation_window` queries
        # only). Mean over (batch, heads, query) gives a per-key-position
        # importance vector of length L_total.
        for idx, layer_attn in enumerate(attentions):
            if idx >= len(self.layer_scores):
                continue
            attn = layer_attn.detach().float()
            observed = attn.mean(dim=(0, 1, 2))
            scores = self.layer_scores[idx]
            key_span = observed.numel()
            if key_span == 0 or key_span > scores.numel():
                continue
            scores = self.score_alpha * scores
            scores[-key_span:] += observed.to(scores.device)
            self.layer_scores[idx] = scores

    def _topk_indices(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        budget = self.retain_count(seq_len)
        if seq_len <= budget:
            return torch.arange(seq_len, device=scores.device)

        sink = min(self.sink_tokens, seq_len)
        recent = min(self.recent_window, max(budget - sink, 0), max(seq_len - sink, 0))
        candidate_end = max(seq_len - recent, sink)
        candidate_scores = scores[sink:candidate_end]
        available = max(budget - sink - recent, 0)
        if available > 0 and candidate_scores.numel() > 0:
            _, topk = torch.topk(candidate_scores, k=min(available, candidate_scores.numel()))
            middle = (topk + sink).sort().values
        else:
            middle = torch.empty(0, device=scores.device, dtype=torch.long)

        prefix = torch.arange(sink, device=scores.device)
        suffix = torch.arange(seq_len - recent, seq_len, device=scores.device) if recent else torch.empty(0, device=scores.device, dtype=torch.long)
        return torch.cat([prefix, middle, suffix]).unique(sorted=True)

    def prune(
        self,
        past_key_values: PastKeyValues | None,
        attentions: tuple[torch.Tensor, ...] | None = None,
    ) -> PastKeyValues | None:
        if not past_key_values:
            return past_key_values
        self._ensure_scores(past_key_values)
        self._resize_scores(past_key_values)

        if attentions is None:
            # Decode step: no fresh observation, no eviction. SnapKV evicts
            # exactly once at end of prefill; growth here is bounded by
            # max_new_tokens which is tiny relative to the pruned budget.
            return past_key_values

        self._observe_attentions(attentions, past_key_values)

        pruned_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        new_scores: list[torch.Tensor] = []
        retained_indices: list[torch.Tensor] = []
        for idx, layer in enumerate(past_key_values):
            seq_len = _seq_len(layer)
            scores = self.layer_scores[idx]
            indices = self._topk_indices(scores, seq_len)
            pruned_layers.append(_slice_layer(layer, indices))
            new_scores.append(scores.index_select(0, indices))
            retained_indices.append(indices.detach().cpu())
        self.layer_scores = new_scores
        self.last_retained_indices = retained_indices
        return tuple(pruned_layers)

    def debug_snapshot(self) -> dict | None:
        if not self.last_retained_indices:
            return None
        first = self.last_retained_indices[0].tolist()
        return {
            "retained_count_layer0": len(first),
            "retained_indices_layer0": first,
        }


@dataclass
class PageHeatPolicy(CachePolicy):
    name: str = "pageheat"
    observation_window: int = 32
    page_size: int = 16
    pin_sink_page: bool = True
    pin_recent_pages: int = 2
    attention_threshold: float = 0.01
    predictor_path: str | None = None
    token_positions: torch.Tensor | None = None
    page_last_attended: dict[int, int] = field(default_factory=dict)
    decode_step: int = 0
    predictor: object | None = None
    last_retained_indices: torch.Tensor | None = None
    last_retained_page_ids: list[int] = field(default_factory=list)
    last_page_scores: dict[int, float] = field(default_factory=dict)

    def requires_attentions(self) -> bool:
        return True

    def requires_decode_attentions(self) -> bool:
        return True

    def _ensure_positions(self, seq_len: int, device: torch.device) -> None:
        if self.token_positions is None:
            self.token_positions = torch.arange(seq_len, device=device, dtype=torch.long)
            return
        if self.token_positions.numel() < seq_len:
            start = int(self.token_positions[-1].item()) + 1
            extra = torch.arange(start, start + (seq_len - self.token_positions.numel()), device=device, dtype=torch.long)
            self.token_positions = torch.cat([self.token_positions.to(device), extra], dim=0)
        elif self.token_positions.numel() > seq_len:
            self.token_positions = self.token_positions[-seq_len:].to(device)
        else:
            self.token_positions = self.token_positions.to(device)

    def _load_predictor(self):
        if self.predictor is not None or not self.predictor_path:
            return
        self.predictor = load_predictor_checkpoint(Path(self.predictor_path))

    def _score_pages(self, batch) -> torch.Tensor:
        self._load_predictor()
        if self.predictor is None:
            recency = 1.0 / (1.0 + batch.age)
            return (0.8 * batch.page_mean_attention) + (0.2 * recency)
        return self.predictor.score(batch.feature_matrix)

    def _pinned_pages(self, page_ids: torch.Tensor) -> set[int]:
        pinned = {0} if self.pin_sink_page else set()
        if page_ids.numel() == 0:
            return pinned
        max_page = int(page_ids.max().item())
        for page_id in range(max(0, max_page - self.pin_recent_pages + 1), max_page + 1):
            pinned.add(page_id)
        return pinned

    def _select_pages(self, batch, budget: int) -> set[int]:
        pinned = self._pinned_pages(batch.page_ids)
        page_counts = {int(page_id): int(count) for page_id, count in zip(batch.page_ids.tolist(), batch.token_counts.tolist())}
        kept = set(page_id for page_id in pinned if page_id in page_counts)
        kept_tokens = sum(page_counts[page_id] for page_id in kept)

        page_scores = self._score_pages(batch)
        self.last_page_scores = {
            int(page_id): float(score)
            for page_id, score in zip(batch.page_ids.tolist(), page_scores.tolist())
        }

        if kept_tokens >= budget:
            return kept

        ranked = sorted(
            (
                (
                    int(page_id),
                    float(score),
                    page_counts[int(page_id)],
                    float(score) / max(page_counts[int(page_id)], 1),
                )
                for page_id, score in zip(batch.page_ids.tolist(), page_scores.tolist())
                if int(page_id) not in kept
            ),
            key=lambda item: (item[3], item[1], -item[2]),
            reverse=True,
        )
        for page_id, _score, token_count, _density in ranked:
            if kept_tokens >= budget:
                break
            if kept_tokens + token_count > budget:
                continue
            kept.add(page_id)
            kept_tokens += token_count
        return kept

    def prune(
        self,
        past_key_values: PastKeyValues | None,
        attentions: tuple[torch.Tensor, ...] | None = None,
    ) -> PastKeyValues | None:
        if not past_key_values:
            return past_key_values

        seq_len = _seq_len(past_key_values[0])
        device = past_key_values[0][0].device
        self._ensure_positions(seq_len, device)

        if attentions is None:
            return past_key_values

        batch = build_page_feature_batch(
            attentions=attentions,
            past_key_values=past_key_values,
            token_positions=self.token_positions,
            page_size=self.page_size,
            page_last_attended=self.page_last_attended,
            decode_step=self.decode_step,
        )
        active_pages = batch.page_ids[batch.page_mean_attention >= self.attention_threshold].tolist()
        for page_id in active_pages:
            self.page_last_attended[int(page_id)] = self.decode_step

        budget = self.retain_count(seq_len)
        kept_pages = self._select_pages(batch, budget)
        page_ids = torch.div(self.token_positions, self.page_size, rounding_mode="floor")
        keep_mask = torch.tensor([int(page_id) in kept_pages for page_id in page_ids.tolist()], device=device, dtype=torch.bool)
        indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return past_key_values

        self.token_positions = self.token_positions.index_select(0, indices)
        self.last_retained_indices = indices.detach().cpu()
        self.last_retained_page_ids = sorted(kept_pages)
        self.decode_step += 1
        return tuple(_slice_layer(layer, indices) for layer in past_key_values)

    def debug_snapshot(self) -> dict | None:
        if self.last_retained_indices is None:
            return None
        return {
            "retained_count_layer0": int(self.last_retained_indices.numel()),
            "retained_indices_layer0": self.last_retained_indices.tolist(),
            "retained_page_ids": self.last_retained_page_ids,
            "page_scores": self.last_page_scores,
        }


def build_cache_policy(
    policy: str,
    retention_rate: float,
    sink_tokens: int,
    recent_window: int,
    observation_window: int,
    page_size: int = 16,
    pin_sink_page: bool = True,
    pin_recent_pages: int = 2,
    pageheat_attention_threshold: float = 0.01,
    pageheat_predictor_path: str | None = None,
) -> CachePolicy:
    policy = policy.lower()
    common = {
        "retention_rate": retention_rate,
        "sink_tokens": sink_tokens,
        "recent_window": recent_window,
    }
    if policy == "full":
        return FullCachePolicy(**common)
    if policy == "streamingllm":
        return StreamingLLMPolicy(**common)
    if policy == "snapkv":
        return SnapKVPolicy(observation_window=observation_window, **common)
    if policy == "pageheat":
        return PageHeatPolicy(
            observation_window=observation_window,
            page_size=page_size,
            pin_sink_page=pin_sink_page,
            pin_recent_pages=pin_recent_pages,
            attention_threshold=pageheat_attention_threshold,
            predictor_path=pageheat_predictor_path,
            **common,
        )
    raise ValueError(f"Unsupported cache policy: {policy}")


def cache_lengths(past_key_values: PastKeyValues | None) -> list[int]:
    if not past_key_values:
        return []
    return [_seq_len(layer) for layer in past_key_values]
