from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.graph.state import GraphState
from src.observability.logging import log_node


def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


class GraphNodes:
    def __init__(self, *, config: AppConfig, llm_client: Any, retriever: Any, reranker: Any, logger: Any):
        self.config = config
        self.llm = llm_client
        self.retriever = retriever
        self.reranker = reranker
        self.logger = logger
        self.router_prompt = _load_prompt("prompts/router.txt")
        self.generator_prompt = _load_prompt("prompts/generator.txt")
        self.self_correction_prompt = _load_prompt("prompts/self_correction.txt")

    def router(self, state: dict) -> dict:
        s = GraphState.model_validate(state)
        start = time.perf_counter()
        try:
            prompt = f"{self.router_prompt}\n\nQUERY:\n{s.query}"
            out = self.llm.invoke_json(prompt, temperature=self.config.llm.router_temperature)
            s.needs_retrieval = bool(out.get("needs_retrieval", True))
            s.rewritten_query = out.get("rewritten_query", s.query)
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            s.status = "fallback"
            s.answer = self.config.fallback_answer
            s.errors.append(f"router_error:{exc}")
            status = "fallback"
        latency_ms = (time.perf_counter() - start) * 1000
        s.latency_ms["router"] = latency_ms
        log_node(self.logger, request_id=s.request_id, node="router", latency_ms=latency_ms, status=status)
        return s.to_dict()

    def retriever_node(self, state: dict) -> dict:
        s = GraphState.model_validate(state)
        start = time.perf_counter()
        try:
            query = s.rewritten_query or s.query
            s.retrieved_chunks = self.retriever.search(query=query, top_k=self.config.qdrant.top_k)
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            s.status = "fallback"
            s.answer = self.config.fallback_answer
            s.errors.append(f"retriever_error:{exc}")
            s.retrieved_chunks = []
            status = "fallback"
        latency_ms = (time.perf_counter() - start) * 1000
        s.latency_ms["retriever"] = latency_ms
        log_node(self.logger, request_id=s.request_id, node="retriever", latency_ms=latency_ms, status=status)
        return s.to_dict()

    def reranker_node(self, state: dict) -> dict:
        s = GraphState.model_validate(state)
        start = time.perf_counter()
        try:
            s.reranked_chunks = self.reranker.rerank(
                query=s.rewritten_query or s.query,
                chunks=s.retrieved_chunks,
                top_k=self.config.qdrant.top_k,
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            s.status = "fallback"
            s.answer = self.config.fallback_answer
            s.errors.append(f"reranker_error:{exc}")
            s.reranked_chunks = s.retrieved_chunks[: self.config.qdrant.top_k]
            status = "fallback"
        latency_ms = (time.perf_counter() - start) * 1000
        s.latency_ms["reranker"] = latency_ms
        log_node(self.logger, request_id=s.request_id, node="reranker", latency_ms=latency_ms, status=status)
        return s.to_dict()

    def generator(self, state: dict) -> dict:
        s = GraphState.model_validate(state)
        start = time.perf_counter()
        try:
            context = "\n".join([c.get("text", "") for c in s.reranked_chunks])
            prompt = (
                f"{self.generator_prompt}\n\nCONTEXT:\n{context}\n\nQUERY:\n{s.query}\n"
                "Return valid JSON only."
            )
            out = self.llm.invoke_json(prompt, temperature=self.config.llm.generator_temperature)
            s.status = out.get("status", "ok")
            s.answer = out.get("answer", "")
            s.citations = out.get("citations", [])
            if s.status == "insufficient_context" and not s.answer:
                s.answer = out.get("missing", "Недостаточно контекста.")
            status = s.status
        except Exception as exc:  # noqa: BLE001
            s.status = "fallback"
            s.answer = self.config.fallback_answer
            s.errors.append(f"generator_error:{exc}")
            status = "fallback"
        latency_ms = (time.perf_counter() - start) * 1000
        s.latency_ms["generator"] = latency_ms
        log_node(self.logger, request_id=s.request_id, node="generator", latency_ms=latency_ms, status=status)
        return s.to_dict()

    def self_correction(self, state: dict) -> dict:
        s = GraphState.model_validate(state)
        start = time.perf_counter()
        try:
            context = "\n".join([c.get("text", "") for c in s.reranked_chunks])
            prompt = (
                f"{self.self_correction_prompt}\n\nQUERY:\n{s.query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{s.answer}"
            )
            out = self.llm.invoke_json(prompt, temperature=0.0)
            s.is_faithful = bool(out.get("is_faithful", True))
            s.covers_query = bool(out.get("covers_query", True))
            s.hallucinated_fragments = out.get("hallucinated_fragments", [])
            s.needs_correction = bool(out.get("needs_correction", False))
            s.correction_query = out.get("correction_query", "")
            if s.needs_correction and s.iteration < s.max_iterations:
                s.iteration += 1
                if s.correction_query:
                    s.query = s.correction_query
                    s.rewritten_query = s.correction_query
            elif s.needs_correction and s.iteration >= s.max_iterations:
                s.status = "low_confidence"
                s.needs_correction = False
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            s.status = "fallback"
            s.needs_correction = False
            s.answer = self.config.fallback_answer
            s.errors.append(f"self_correction_error:{exc}")
            status = "fallback"
        latency_ms = (time.perf_counter() - start) * 1000
        s.latency_ms["self_correction"] = latency_ms
        log_node(
            self.logger,
            request_id=s.request_id,
            node="self_correction",
            latency_ms=latency_ms,
            status=status,
        )
        return s.to_dict()

    def output(self, state: dict) -> dict:
        s = GraphState.model_validate(state)
        total = sum(s.latency_ms.values())
        s.latency_ms["total"] = total
        if not s.answer and s.status in {"fallback", "low_confidence"}:
            s.answer = self.config.fallback_answer
        return s.to_dict()


def route_decision(state: dict) -> str:
    s = GraphState.model_validate(state)
    if s.status == "fallback":
        return "output"
    return "retriever" if s.needs_retrieval else "output"


def correction_decision(state: dict) -> str:
    s = GraphState.model_validate(state)
    if s.status == "fallback":
        return "output"
    if s.needs_correction and s.iteration < s.max_iterations:
        return "retriever"
    return "output"
