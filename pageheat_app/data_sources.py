from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Sample:
    sample_id: str
    prompt: str
    target: str
    task: str
    metric: str
    metadata: dict[str, Any]


def extract_prompt_and_target(record: dict[str, Any], prompt_field: str, target_field: str) -> tuple[str, str]:
    prompt = str(record.get(prompt_field, "")).strip()
    target = str(record.get(target_field, "")).strip()
    return prompt, target

