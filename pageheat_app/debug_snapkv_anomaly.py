from __future__ import annotations

import argparse
import json
from pathlib import Path

from pageheat_app.utils import read_json


def load_predictions(path: Path) -> list[dict]:
    payload = read_json(path)
    return payload["predictions"]


def compare_reports(path_20: Path, path_50: Path, top_k: int = 10) -> dict:
    preds_20 = load_predictions(path_20)
    preds_50 = load_predictions(path_50)
    rows = []
    for item20, item50 in zip(preds_20, preds_50):
        delta = item50["score"] - item20["score"]
        rows.append(
            {
                "sample_index": item20["sample_index"],
                "delta_50_minus_20": delta,
                "score_20": item20["score"],
                "score_50": item50["score"],
                "target": item20["target"],
                "prediction_20": item20["prediction"],
                "prediction_50": item50["prediction"],
                "debug_20": item20.get("debug"),
                "debug_50": item50.get("debug"),
            }
        )
    rows.sort(key=lambda item: item["delta_50_minus_20"])
    return {
        "worse_50_than_20_count": sum(1 for row in rows if row["delta_50_minus_20"] < 0),
        "better_50_than_20_count": sum(1 for row in rows if row["delta_50_minus_20"] > 0),
        "same_count": sum(1 for row in rows if row["delta_50_minus_20"] == 0),
        "worst_regressions": rows[:top_k],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare SnapKV 20% and 50% reports to inspect non-monotonic regressions.")
    parser.add_argument("--report-20", type=Path, required=True)
    parser.add_argument("--report-50", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    result = compare_reports(args.report_20, args.report_50, args.top_k)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
