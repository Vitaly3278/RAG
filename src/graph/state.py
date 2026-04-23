from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


StatusType = Literal["ok", "insufficient_context", "low_confidence", "fallback"]


class GraphState(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    needs_retrieval: bool = True
    rewritten_query: str = ""
    retrieved_chunks: list[dict] = Field(default_factory=list)
    reranked_chunks: list[dict] = Field(default_factory=list)
    answer: str = ""
    citations: list[dict] = Field(default_factory=list)
    status: StatusType = "ok"
    hallucinated_fragments: list[str] = Field(default_factory=list)
    covers_query: bool = True
    is_faithful: bool = True
    needs_correction: bool = False
    correction_query: str = ""
    iteration: int = 0
    max_iterations: int = 2
    latency_ms: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()
