from __future__ import annotations

import json
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from src.config import load_config
from src.graph.nodes import GraphNodes
from src.graph.workflow import RAGService
from src.llm.client import LLMClient
from src.observability.logging import setup_logging
from src.observability.tracing import setup_tracing
from src.retrieval.qdrant_client import QdrantRetriever
from src.retrieval.reranker import BGEReranker


REQUEST_COUNT = Counter("rag_requests_total", "Total RAG requests")
REQUEST_LATENCY = Histogram("rag_request_latency_seconds", "RAG request latency")


def default_backend(prompt: str, **kwargs) -> str:  # noqa: ANN003
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


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    request_id: str | None = None


class AskResponse(BaseModel):
    request_id: str
    status: str
    answer: str
    citations: list[dict]
    iteration_count: int
    needs_retrieval: bool
    timings_ms: dict[str, float]


def create_app() -> FastAPI:
    cfg = load_config()
    logger = setup_logging()
    setup_tracing(service_name=cfg.service_name)

    llm = LLMClient(default_backend, retries=cfg.llm.json_retry_attempts)
    try:
        retriever = QdrantRetriever(
            url=cfg.qdrant.url,
            collection=cfg.qdrant.collection,
            timeout_seconds=cfg.qdrant.timeout_seconds,
        )
    except Exception:  # noqa: BLE001
        class SafeRetriever:
            def search(self, query: str, top_k: int = 5):  # noqa: ANN001
                return []

        retriever = SafeRetriever()
    reranker = BGEReranker()
    nodes = GraphNodes(config=cfg, llm_client=llm, retriever=retriever, reranker=reranker, logger=logger)
    service = RAGService(app_config=cfg, nodes=nodes)

    app = FastAPI(title="ContextGuard RAG Service")

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        REQUEST_COUNT.inc()
        started = perf_counter()
        request_id = payload.request_id or str(uuid4())
        result = service.run(payload.query, request_id=request_id)
        REQUEST_LATENCY.observe(perf_counter() - started)

        return AskResponse(
            request_id=result.request_id,
            status=result.status,
            answer=result.answer,
            citations=result.citations,
            iteration_count=result.iteration,
            needs_retrieval=result.needs_retrieval,
            timings_ms={
                "router": result.latency_ms.get("router", 0.0),
                "retriever": result.latency_ms.get("retriever", 0.0),
                "reranker": result.latency_ms.get("reranker", 0.0),
                "generator": result.latency_ms.get("generator", 0.0),
                "self_correction": result.latency_ms.get("self_correction", 0.0),
                "total": result.latency_ms.get("total", 0.0),
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
