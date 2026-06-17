from __future__ import annotations

import os
import pathlib
import sys
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class APIConfig(BaseModel):
    """OpenRouter-compatible API configuration."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout: float = 120.0


class OCRConfig(BaseModel):
    """Mechanical OCR engine configuration."""

    engine: Literal["rapidocr", "pytesseract"] = "rapidocr"
    # Only used by tesseract. Examples: "eng", "jpn", "eng+jpn".
    languages: list[str] = Field(default_factory=lambda: ["eng", "jpn"])


class ModelConfig(BaseModel):
    """LLM model configuration."""

    mode: Literal["multimodal", "hybrid"] = "multimodal"
    primary_model: str = "google/gemini-2.0-flash-001"
    vision_model: Optional[str] = "openai/gpt-4o-mini"
    temperature: float = 0.2
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class AgentConfig(BaseModel):
    """Agent loop configuration."""

    max_iterations: int = Field(default=3, ge=1, le=10)
    extract_system: Optional[str] = None
    integrate_system: Optional[str] = None
    critique_system: Optional[str] = None


class Config(BaseModel):
    api: APIConfig = Field(default_factory=APIConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(
    path: Optional[pathlib.Path] = None, require_api_key: bool = True
) -> tuple[Config, Optional[str]]:
    """Load TOML config and read the API key from the configured env var."""
    path = path or pathlib.Path.home() / ".config" / "aiocr" / "config.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            "Create one at ~/.config/aiocr/config.toml or pass --config."
        )

    with open(path, "rb") as f:
        data = tomllib.load(f)

    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid config: {exc}") from exc

    if config.model.mode == "hybrid" and not config.model.vision_model:
        raise RuntimeError(
            "hybrid mode requires model.vision_model in config.toml"
        )

    api_key = os.environ.get(config.api.api_key_env)
    if require_api_key and not api_key:
        raise RuntimeError(
            f"Environment variable {config.api.api_key_env!r} is not set. "
            "Set it with the OpenRouter-compatible API key."
        )

    return config, api_key


SAMPLE_CONFIG = """# aiocr configuration sample
# Place this at ~/.config/aiocr/config.toml
# All fields have defaults — only override what you want to change.

[api]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
timeout = 120

[ocr]
engine = "rapidocr"        # "rapidocr" or "pytesseract"
languages = ["eng", "jpn"] # only meaningful for tesseract

[model]
mode = "multimodal"        # "multimodal" or "hybrid"
primary_model = "google/gemini-2.0-flash-001"
# vision_model = "openai/gpt-4o-mini"  # used in hybrid mode
temperature = 0.2
max_tokens = 8192

[agent]
max_iterations = 3
"""
