from __future__ import annotations

from pathlib import Path

import modal
import torch

from pageheat_app.build_agent_dataset import build_dataset
from pageheat_app.collect_pageheat_data import collect_dataset
from pageheat_app.eval_harness import EvalArgs, run_and_save_eval
from pageheat_app.run_baselines import run_sweep
from pageheat_app.settings import DATA_DIR
from pageheat_app.smoke import run_smoke_test


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch==2.4.1",
        "transformers==4.45.2",
        "datasets==2.21.0",
        "accelerate==0.34.2",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "evaluate==0.4.3",
        "rouge-score==0.1.2",
        "matplotlib==3.9.2",
        "psutil==6.0.0",
        "tqdm==4.66.5",
        "numpy==2.1.1",
        "pandas==2.2.2",
        "pyyaml==6.0.2",
    )
    .pip_install("flash-attn==2.6.3", extra_options="--no-build-isolation")
    .add_local_python_source("pageheat_app")
)

app = modal.App("pageheat-week1", image=image)


@app.function(gpu="A100-40GB", timeout=60 * 60)
def qwen_smoke_test(attn_implementation: str = "flash_attention_2"):
    try:
        return run_smoke_test(attn_implementation)
    except Exception:
        if attn_implementation == "flash_attention_2":
            return run_smoke_test("sdpa")
        raise


@app.function(gpu="A100-40GB", timeout=8 * 60 * 60)
def remote_eval(
    eval_args: dict,
    agent_config_text: str | None = None,
    predictor_bytes: bytes | None = None,
):
    if isinstance(eval_args.get("results_dir"), str):
        eval_args["results_dir"] = Path(eval_args["results_dir"])
    if eval_args.get("dataset") == "agent_traces" and agent_config_text:
        agent_config_path = Path("/tmp/pageheat_agent_sources.yaml")
        agent_config_path.write_text(agent_config_text, encoding="utf-8")
        build_dataset(agent_config_path, DATA_DIR / "agent_traces")
    if predictor_bytes is not None:
        predictor_path = Path("/tmp/pageheat_predictor.pt")
        predictor_path.write_bytes(predictor_bytes)
        eval_args["pageheat_predictor_path"] = str(predictor_path)
    args = EvalArgs(**eval_args)
    report, output_path = run_and_save_eval(args)
    return {
        "summary": report["summary"],
        "report": report,
        "saved_report": str(output_path),
    }


@app.function(gpu="A100-40GB", timeout=24 * 60 * 60)
def remote_run_sweep(
    config_text: str,
    agent_config_text: str | None = None,
    predictor_bytes: bytes | None = None,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    attn_implementation: str = "flash_attention_2",
    results_dir: str = "/tmp/pageheat_results",
    sink_tokens: int = 4,
    recent_window: int = 512,
    observation_window: int = 32,
    page_size: int = 16,
    pin_sink_page: bool = True,
    pin_recent_pages: int = 2,
    pageheat_attention_threshold: float = 0.01,
    pageheat_predictor_path: str | None = None,
):
    config_path = Path("/tmp/pageheat_sweep_config.yaml")
    config_path.write_text(config_text, encoding="utf-8")
    if agent_config_text:
        agent_config_path = Path("/tmp/pageheat_agent_sources.yaml")
        agent_config_path.write_text(agent_config_text, encoding="utf-8")
        build_dataset(agent_config_path, DATA_DIR / "agent_traces")
    remote_predictor_path = None
    if predictor_bytes is not None:
        predictor_path = Path("/tmp/pageheat_predictor.pt")
        predictor_path.write_bytes(predictor_bytes)
        remote_predictor_path = str(predictor_path)
    return run_sweep(
        config_path=config_path,
        model_id=model_id,
        attn_implementation=attn_implementation,
        results_dir=Path(results_dir),
        sink_tokens=sink_tokens,
        recent_window=recent_window,
        observation_window=observation_window,
        page_size=page_size,
        pin_sink_page=pin_sink_page,
        pin_recent_pages=pin_recent_pages,
        pageheat_attention_threshold=pageheat_attention_threshold,
        pageheat_predictor_path=remote_predictor_path or pageheat_predictor_path,
    )


@app.function(gpu="A100-40GB", timeout=24 * 60 * 60)
def remote_collect_pageheat_data(collect_args: dict, agent_config_text: str | None = None):
    if isinstance(collect_args.get("output"), str):
        collect_args["output"] = Path(collect_args["output"])
    if collect_args.get("dataset") == "agent_traces" and agent_config_text:
        agent_config_path = Path("/tmp/pageheat_agent_sources.yaml")
        agent_config_path.write_text(agent_config_text, encoding="utf-8")
        build_dataset(agent_config_path, DATA_DIR / "agent_traces")
    args = type("CollectArgs", (), collect_args)
    summary = collect_dataset(args)
    payload = torch.load(args.output, map_location="cpu")
    return {
        "summary": summary,
        "payload": payload,
        "saved_dataset": str(args.output),
    }


@app.local_entrypoint()
def main(command: str = "smoke", attn_implementation: str = "flash_attention_2"):
    if command == "smoke":
        print(qwen_smoke_test.remote(attn_implementation))
        return
    raise SystemExit(f"Unsupported command: {command}")
