from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import yaml
from datasets import Dataset, DatasetDict

from pageheat_app.settings import DATA_DIR
from pageheat_app.utils import ensure_dir, write_json


def _conversation_to_prompt(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _extract_sample(record: dict[str, Any], source_name: str, prompt_key: str, target_key: str, task: str, metric: str) -> dict[str, Any] | None:
    if prompt_key == "__history__":
        history = record.get("history") or record.get("messages") or []
        prompt = _conversation_to_prompt(history[:-1]) if len(history) > 1 else _conversation_to_prompt(history)
        target = history[-1].get("content", "") if history else ""
    else:
        prompt = str(record.get(prompt_key, "")).strip()
        target = str(record.get(target_key, "")).strip()

    if not prompt or not target:
        return None

    return {
        "source": source_name,
        "task": task,
        "metric": metric,
        "prompt": prompt,
        "target": target,
        "prompt_chars": len(prompt),
        "target_chars": len(target),
    }


def _render_messages(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content")
        if content is not None:
            rendered.append(f"{role}: {content}")
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            rendered.append(f"{role}_tool_calls: {json.dumps(tool_calls, ensure_ascii=True, sort_keys=True)}")
    return "\n".join(rendered).strip()


def _load_remote_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def _load_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    source_type = source.get("type", "hf_dataset")
    if source_type == "json_url":
        text = _load_remote_text(source["url"])
        file_format = source.get("format", "json")
        if file_format == "json":
            payload = json.loads(text)
            if isinstance(payload, list):
                return payload
            raise ValueError(f"Expected a JSON array from {source['url']}")
        if file_format == "jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        raise ValueError(f"Unsupported format: {file_format}")
    raise ValueError(f"Unsupported source type: {source_type}")


def _extract_tau_bench_pairs(record: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for turn in record.get("traj", []):
        role = turn.get("role")
        target = _render_messages([turn])
        if role == "assistant" and history and target:
            prompt = _render_messages(history)
            if prompt:
                samples.append(
                    {
                        "source": source["name"],
                        "task": source["task"],
                        "metric": source["metric"],
                        "prompt": prompt,
                        "target": target,
                        "prompt_chars": len(prompt),
                        "target_chars": len(target),
                    }
                )
        history.append(turn)
    return samples


def _extract_bfcl_pairs(record: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    question_turns = record.get("question", [])
    path = record.get("path", [])
    history: list[dict[str, Any]] = []

    for idx, turn_messages in enumerate(question_turns):
        history.extend(turn_messages)
        if idx >= len(path):
            break
        prompt = _render_messages(history)
        target = str(path[idx]).strip()
        if not prompt or not target:
            continue
        samples.append(
            {
                "source": source["name"],
                "task": source["task"],
                "metric": source["metric"],
                "prompt": prompt,
                "target": target,
                "prompt_chars": len(prompt),
                "target_chars": len(target),
            }
        )
    return samples


def _extract_samples_from_record(record: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    extractor = source.get("extractor", "simple_fields")
    if extractor == "tau_bench_pairs":
        return _extract_tau_bench_pairs(record, source)
    if extractor == "bfcl_pairs":
        return _extract_bfcl_pairs(record, source)

    sample = _extract_sample(
        record=record,
        source_name=source["name"],
        prompt_key=source["prompt_key"],
        target_key=source.get("target_key", ""),
        task=source["task"],
        metric=source["metric"],
    )
    return [sample] if sample is not None else []


def build_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"sources": {}}

    for source in config["sources"]:
        records = _load_records(source)
        max_samples = int(source.get("max_samples", len(records)))
        kept = 0
        for record in records:
            for sample in _extract_samples_from_record(record, source):
                samples.append(sample)
                kept += 1
                if kept >= max_samples:
                    break
            if kept >= max_samples:
                break
        stats["sources"][source["name"]] = {"kept_samples": kept}

    dataset = Dataset.from_list(samples)
    dataset_dict = DatasetDict({"train": dataset})
    ensure_dir(output_dir)
    dataset_dict.save_to_disk(str(output_dir))

    prompt_lengths = [sample["prompt_chars"] for sample in samples]
    stats["num_samples"] = len(samples)
    stats["mean_prompt_chars"] = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
    stats["max_prompt_chars"] = max(prompt_lengths) if prompt_lengths else 0
    stats["min_prompt_chars"] = min(prompt_lengths) if prompt_lengths else 0
    write_json(output_dir / "stats.json", stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Week 1 agent-trace dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "agent_traces")
    args = parser.parse_args(argv)

    stats = build_dataset(args.config, args.output_dir)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
