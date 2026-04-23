from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from src.config import AppConfig
from src.graph.nodes import GraphNodes
from src.graph.workflow import RAGService
from src.llm.client import LLMClient
from src.llm.ollama_backend import OllamaBackend
from src.observability.logging import setup_logging
from src.retrieval.qdrant_client import QdrantRetriever
from src.retrieval.reranker import BGEReranker


def _default_stub_backend(prompt: str, **kwargs: Any) -> str:  # noqa: ANN401
    prompt_lower = prompt.lower()
    if '"needs_retrieval"' in prompt or "router" in prompt_lower:
        return json.dumps({"needs_retrieval": True, "rewritten_query": " ".join(prompt.split()[-12:])})
    if '"is_faithful"' in prompt or "self-correction" in prompt_lower:
        return json.dumps(
            {
                "is_faithful": True,
                "covers_query": True,
                "hallucinated_fragments": [],
                "needs_correction": False,
                "correction_query": "",
            }
        )
    return json.dumps(
        {
            "status": "ok",
            "answer": "Ответ сформирован на основе доступного контекста.",
            "citations": [],
            "missing": "",
        }
    )


def build_llm_client(config: AppConfig) -> LLMClient:
    provider = os.getenv("RAG_LLM_PROVIDER", config.llm.provider).lower()
    if provider == "ollama" and _is_ollama_available(config.llm.ollama_url):
        backend = OllamaBackend(
            base_url=config.llm.ollama_url,
            model=config.llm.model_name,
            timeout_seconds=config.llm.timeout_seconds,
        )
    else:
        backend = _default_stub_backend
    return LLMClient(backend, retries=config.llm.json_retry_attempts)


def _is_ollama_available(base_url: str) -> bool:
    req = request.Request(url=f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with request.urlopen(req, timeout=0.4):
            return True
    except (TimeoutError, error.URLError):
        return False


def build_rag_service(config: AppConfig) -> RAGService:
    logger = setup_logging()
    llm = build_llm_client(config)
    try:
        retriever = QdrantRetriever(
            url=config.qdrant.url,
            collection=config.qdrant.collection,
            timeout_seconds=config.qdrant.timeout_seconds,
            vector_size=config.qdrant.vector_size,
        )
    except Exception:  # noqa: BLE001
        class SafeRetriever:
            def search(self, query: str, top_k: int = 5):  # noqa: ANN001
                return []

        retriever = SafeRetriever()
    reranker = BGEReranker()
    nodes = GraphNodes(config=config, llm_client=llm, retriever=retriever, reranker=reranker, logger=logger)
    return RAGService(app_config=config, nodes=nodes)
