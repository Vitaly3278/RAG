from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BGEReranker:
    """
    Lightweight placeholder for BGE reranker integration.
    Replace score logic with actual model inference in production.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        # Deterministic fallback rerank: keep existing score ordering.
        ordered = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        return ordered[:top_k]
