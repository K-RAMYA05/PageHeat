from __future__ import annotations

import re
from typing import Iterable


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_match(prediction: str, target: str) -> float:
    return float(normalize_text(prediction) == normalize_text(target))


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    overlap = 0
    for token in target_tokens:
        count = pred_counts.get(token, 0)
        if count:
            overlap += 1
            pred_counts[token] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_mcq_letter(text: str) -> str:
    if isinstance(text, int):
        return {0: "A", 1: "B", 2: "C", 3: "D"}.get(text, "")
    if isinstance(text, float) and text.is_integer():
        return {0: "A", 1: "B", 2: "C", 3: "D"}.get(int(text), "")

    normalized = str(text).strip().upper()
    if normalized in {"0", "1", "2", "3"}:
        return {"0": "A", "1": "B", "2": "C", "3": "D"}[normalized]
    patterns = [
        r"THE CORRECT ANSWER IS\s*\(?([ABCD])\)?",
        r"ANSWER\s*[:\-]?\s*\(?([ABCD])\)?",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return normalized[:1] if normalized[:1] in {"A", "B", "C", "D"} else ""


def mcq_exact_match(prediction: str, target: str) -> float:
    return float(extract_mcq_letter(prediction) == extract_mcq_letter(target))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
