from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from src.bootstrap import build_rag_service
from src.config import load_config
from src.observability.tracing import setup_tracing


REQUEST_COUNT = Counter("rag_requests_total", "Total RAG requests")
REQUEST_LATENCY = Histogram("rag_request_latency_seconds", "RAG request latency")


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
    setup_tracing(service_name=cfg.service_name)
    service = build_rag_service(cfg)

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
