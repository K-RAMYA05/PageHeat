from pathlib import Path

from pageheat_app.failure_analysis import analyze_failures
from pageheat_app.summarize_results import build_summary
from pageheat_app.utils import write_json



def _report(dataset: str, policy: str, retention_rate: float, accuracy: float, tok_s: float, predictions: list[dict]):
    return {
        "summary": {
            "dataset": dataset,
            "accuracy": accuracy,
            "decode_tokens_per_s": tok_s,
            "ttft_s": 0.1,
            "peak_memory_gb": 1.0,
            "cache_policy": {
                "policy": policy,
                "retention_rate": retention_rate,
            },
        },
        "predictions": predictions,
    }


def test_build_summary_computes_pageheat_headline(tmp_path: Path):
    write_json(
        tmp_path / "full.json",
        _report("agent_traces", "full", 1.0, 0.80, 10.0, []),
    )
    write_json(
        tmp_path / "snapkv.json",
        _report("agent_traces", "snapkv", 0.2, 0.70, 14.0, []),
    )
    write_json(
        tmp_path / "pageheat.json",
        _report("agent_traces", "pageheat", 0.2, 0.76, 18.0, []),
    )

    summary = build_summary(tmp_path)

    assert summary["headline"][0]["pageheat_minus_snapkv"] == 0.06
    assert summary["headline"][0]["pageheat_accuracy_vs_full"] == 0.95


def test_analyze_failures_finds_pageheat_regressions(tmp_path: Path):
    shared_predictions = [
        {
            "sample_index": 0,
            "task": "tool_call",
            "target": "A",
            "prediction": "A",
            "score": 1.0,
            "cache_trace": [[20000]],
        },
        {
            "sample_index": 1,
            "task": "tool_call",
            "target": "B",
            "prediction": "A",
            "score": 0.0,
            "cache_trace": [[5000]],
        },
    ]
    pageheat_predictions = [dict(row) for row in shared_predictions]
    snapkv_predictions = [dict(row) for row in shared_predictions]
    full_predictions = [dict(row) for row in shared_predictions]
    snapkv_predictions[1]["score"] = 1.0
    snapkv_predictions[1]["prediction"] = "B"

    write_json(tmp_path / "full.json", _report("agent_traces", "full", 1.0, 0.5, 10.0, full_predictions))
    write_json(tmp_path / "snapkv.json", _report("agent_traces", "snapkv", 0.2, 1.0, 14.0, snapkv_predictions))
    write_json(tmp_path / "pageheat.json", _report("agent_traces", "pageheat", 0.2, 0.5, 18.0, pageheat_predictions))

    summary = analyze_failures(tmp_path, dataset="agent_traces", retention_rate=0.2)

    assert summary["num_failures"] == 1
    assert summary["worst_cases"][0]["sample_index"] == 1
    assert summary["task_summary"][0]["task"] == "tool_call"
