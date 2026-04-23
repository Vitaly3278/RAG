from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval.embeddings import hash_embed


@dataclass
class QdrantRetriever:
    url: str
    collection: str
    timeout_seconds: float = 2.0
    vector_size: int = 256

    def __post_init__(self) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=self.url, timeout=self.timeout_seconds)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = hash_embed(query, dim=self.vector_size)
        points = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        # qdrant-client can return QueryResponse with `points`.
        result = []
        for idx, p in enumerate(getattr(points, "points", []) or []):
            payload = getattr(p, "payload", {}) or {}
            result.append(
                {
                    "id": str(getattr(p, "id", idx)),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "score": float(getattr(p, "score", 0.0) or 0.0),
                }
            )
        return result
