from pathlib import Path

from pageheat_app.decision_check import evaluate_snapkv_gap
from pageheat_app.utils import write_json


def test_decision_check_detects_gap(tmp_path: Path):
    payload = {
        "runs": [
            {"dataset": "agent_traces", "accuracy": 0.8, "cache_policy": {"policy": "full", "retention_rate": 1.0}},
            {"dataset": "agent_traces", "accuracy": 0.6, "cache_policy": {"policy": "snapkv", "retention_rate": 0.1}},
        ]
    }
    write_json(tmp_path / "summary.json", payload)
    result = evaluate_snapkv_gap(tmp_path)
    assert result["verdict"] == "gap_exists"


def test_decision_check_detects_pivot_condition(tmp_path: Path):
    payload = {
        "runs": [
            {"dataset": "agent_traces", "accuracy": 0.8, "cache_policy": {"policy": "full", "retention_rate": 1.0}},
            {"dataset": "agent_traces", "accuracy": 0.78, "cache_policy": {"policy": "snapkv", "retention_rate": 0.1}},
        ]
    }
    write_json(tmp_path / "summary.json", payload)
    result = evaluate_snapkv_gap(tmp_path)
    assert result["verdict"] == "pivot_or_refine_benchmark"


def test_decision_check_flags_non_monotonic_snapkv_accuracy(tmp_path: Path):
    payload = {
        "runs": [
            {"dataset": "agent_traces", "accuracy": 0.8, "cache_policy": {"policy": "full", "retention_rate": 1.0}},
            {"dataset": "agent_traces", "accuracy": 0.78, "cache_policy": {"policy": "snapkv", "retention_rate": 0.2}},
            {"dataset": "agent_traces", "accuracy": 0.76, "cache_policy": {"policy": "snapkv", "retention_rate": 0.5}},
        ]
    }
    write_json(tmp_path / "summary.json", payload)
    result = evaluate_snapkv_gap(tmp_path)
    assert result["verdict"] == "gap_exists"
    assert result["monotonicity_violations"] == [
        {
            "lower_retention_rate": 0.2,
            "lower_retention_accuracy": 0.78,
            "higher_retention_rate": 0.5,
            "higher_retention_accuracy": 0.76,
        }
    ]
