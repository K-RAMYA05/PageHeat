import torch

from pageheat_app.cache_policies import SnapKVPolicy, StreamingLLMPolicy


def _mock_past(seq_len: int, layers: int = 2, heads: int = 2, dim: int = 4):
    past = []
    for _ in range(layers):
        key = torch.randn(1, heads, seq_len, dim)
        value = torch.randn(1, heads, seq_len, dim)
        past.append((key, value))
    return tuple(past)


def test_streamingllm_keeps_sink_and_recent():
    policy = StreamingLLMPolicy(retention_rate=0.5, sink_tokens=2, recent_window=3)
    pruned = policy.prune(_mock_past(seq_len=10))
    assert pruned is not None
    assert pruned[0][0].shape[-2] == 5


def test_snapkv_prunes_to_budget():
    past = _mock_past(seq_len=12)
    attentions = tuple(torch.rand(1, 2, 4, 12) for _ in range(len(past)))
    policy = SnapKVPolicy(retention_rate=0.5, sink_tokens=2, recent_window=2, observation_window=4)
    pruned = policy.prune(past, attentions)
    assert pruned is not None
    assert pruned[0][0].shape[-2] <= 6
