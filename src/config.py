from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"
    timeout_seconds: float = 2.0
    top_k: int = 5


class LLMConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-8B-Instruct"
    timeout_seconds: float = 5.0
    router_temperature: float = 0.1
    generator_temperature: float = 0.3
    evaluator_temperature: float = 0.0
    json_retry_attempts: int = 2


class AppConfig(BaseModel):
    service_name: str = "contextguard-rag"
    max_iterations: int = 2
    cache_ttl_hours: int = 24
    fallback_answer: str = "Сервис временно недоступен, используйте ручную проверку источников."
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw or {})


def load_config(path: str | Path = "configs/settings.yaml") -> AppConfig:
    return AppConfig.from_yaml(path)


def merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(patch)
    return out
