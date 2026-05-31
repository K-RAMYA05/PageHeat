from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
DATA_DIR = ARTIFACTS_DIR / "data"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROMPT = "Explain why FlashAttention matters for throughput and memory efficiency in LLM inference."


@dataclass(slots=True)
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 64
    max_prompt_tokens: int | None = None
    trust_remote_code: bool = True


@dataclass(slots=True)
class CacheConfig:
    policy: str = "full"
    retention_rate: float = 1.0
    sink_tokens: int = 4
    recent_window: int = 512
    observation_window: int = 32
    page_size: int = 16
    pin_sink_page: bool = True
    pin_recent_pages: int = 2
    pageheat_attention_threshold: float = 0.01
    pageheat_predictor_path: str | None = None
