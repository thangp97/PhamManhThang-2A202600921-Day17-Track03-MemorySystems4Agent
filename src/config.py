from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig

import os

@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Bonus knobs (xem ANALYSIS.md mục Bonus):
    # - confidence_threshold: chỉ ghi fact vào User.md khi đủ chắc chắn.
    # - decay_rate: hệ số giảm độ tin cậy của fact theo số lượt không được nhắc lại.
    confidence_threshold: float = 0.6
    decay_rate: float = 0.9


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    provider = os.environ.get("LLM_PROVIDER", "openai")
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    # Mỗi provider lấy api_key từ một biến env khác nhau.
    key_env = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }.get(provider, "OPENAI_API_KEY")
    api_key = os.environ.get(key_env)

    # base_url chỉ cần cho custom (OpenAI-compatible) và ollama.
    if provider == "custom":
        base_url = os.environ.get("CUSTOM_BASE_URL")
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL")
    else:
        base_url = None

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    judge_model = model  # judge dùng chung config với model chính

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=800,
        compact_keep_messages=4,
        model=model,
        judge_model=judge_model,
        confidence_threshold=float(os.environ.get("MEMORY_CONFIDENCE_THRESHOLD", "0.6")),
        decay_rate=float(os.environ.get("MEMORY_DECAY_RATE", "0.9")),
    )
