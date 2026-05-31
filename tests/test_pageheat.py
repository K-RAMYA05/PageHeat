import torch

from pageheat_app.cache_policies import PageHeatPolicy
from pageheat_app.pageheat import (
    PageHeatPredictor,
    build_page_feature_batch,
    choose_hidden_dims_for_target_params,
    predictor_parameter_count,
)


def _mock_past(seq_len: int, layers: int = 2, heads: int = 2, dim: int = 4):
    past = []
    for _ in range(layers):
        key = torch.randn(1, heads, seq_len, dim)
        value = torch.randn(1, heads, seq_len, dim)
        past.append((key, value))
    return tuple(past)


def test_build_page_feature_batch_shapes():
    seq_len = 32
    past = _mock_past(seq_len=seq_len)
    attentions = tuple(torch.rand(1, 2, 4, seq_len) for _ in range(len(past)))
    token_positions = torch.arange(seq_len)

    batch = build_page_feature_batch(
        attentions=attentions,
        past_key_values=past,
        token_positions=token_positions,
        page_size=16,
    )

    assert batch.feature_matrix.shape[0] == 2
    assert batch.page_ids.tolist() == [0, 1]
    assert len(batch.feature_names) == batch.feature_matrix.shape[1]


def test_pageheat_policy_pins_sink_and_recent_pages():
    seq_len = 64
    past = _mock_past(seq_len=seq_len)
    attentions = tuple(torch.rand(1, 2, 4, seq_len) for _ in range(len(past)))
    policy = PageHeatPolicy(retention_rate=0.5, sink_tokens=4, recent_window=16, page_size=16, pin_recent_pages=2)

    pruned = policy.prune(past, attentions)

    assert pruned is not None
    assert 0 in policy.last_retained_page_ids
    assert max(policy.last_retained_page_ids) == 3
    assert 2 in policy.last_retained_page_ids


def test_pageheat_policy_respects_budget_when_selecting_pages():
    seq_len = 65
    past = _mock_past(seq_len=seq_len)
    attentions = tuple(torch.rand(1, 2, 4, seq_len) for _ in range(len(past)))
    policy = PageHeatPolicy(
        retention_rate=0.3,
        sink_tokens=4,
        recent_window=16,
        page_size=16,
        pin_recent_pages=1,
    )

    pruned = policy.prune(past, attentions)

    assert pruned is not None
    assert pruned[0][0].shape[-2] <= policy.retain_count(seq_len)


def test_choose_hidden_dims_for_target_params_hits_budget_band():
    hidden_dims = choose_hidden_dims_for_target_params(
        input_dim=817,
        target_params=500_000,
        num_hidden_layers=2,
    )

    model = PageHeatPredictor(input_dim=817, hidden_dims=hidden_dims)
    param_count = predictor_parameter_count(model)

    assert 425_000 <= param_count <= 575_000
