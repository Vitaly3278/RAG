from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"
    timeout_seconds: float = 2.0
    top_k: int = 5
    vector_size: int = 256


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model_name: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    timeout_seconds: float = 30.0
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
    cfg = AppConfig.from_yaml(path)
    if os.getenv("RAG_QDRANT_URL"):
        cfg.qdrant.url = os.environ["RAG_QDRANT_URL"]
    if os.getenv("RAG_QDRANT_COLLECTION"):
        cfg.qdrant.collection = os.environ["RAG_QDRANT_COLLECTION"]
    if os.getenv("RAG_LLM_PROVIDER"):
        cfg.llm.provider = os.environ["RAG_LLM_PROVIDER"]
    if os.getenv("RAG_LLM_MODEL"):
        cfg.llm.model_name = os.environ["RAG_LLM_MODEL"]
    if os.getenv("RAG_LLM_OLLAMA_URL"):
        cfg.llm.ollama_url = os.environ["RAG_LLM_OLLAMA_URL"]
    if os.getenv("RAG_LLM_TIMEOUT_SECONDS"):
        cfg.llm.timeout_seconds = float(os.environ["RAG_LLM_TIMEOUT_SECONDS"])
    return cfg


def merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(patch)
    return out
