from __future__ import annotations

from src.config import AppConfig
from src.graph.nodes import GraphNodes
from src.graph.workflow import RAGService
from src.llm.client import LLMClient


class FakeRetriever:
    def search(self, query: str, top_k: int = 5):  # noqa: ANN001
        return [
            {"id": "1", "text": "Policy says 10 days", "source": "policy.md", "score": 0.8},
            {"id": "2", "text": "Another chunk", "source": "faq.md", "score": 0.4},
        ][:top_k]


class FakeReranker:
    def rerank(self, query: str, chunks: list[dict], top_k: int = 5):  # noqa: ANN001
        return sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]


class FakeLogger:
    def info(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


def backend(prompt: str, **kwargs):  # noqa: ANN003
    low = prompt.lower()
    if "интеллектуальный маршрутизатор" in low:
        return '{"needs_retrieval": true, "rewritten_query": "policy vacation days"}'
    if "строгий аудитор" in low:
        return (
            '{"is_faithful": true, "covers_query": true, "hallucinated_fragments": [], '
            '"needs_correction": false, "correction_query": ""}'
        )
    return (
        '{"status":"ok","answer":"Согласно policy, отпуск 10 дней.",'
        '"citations":[{"id":"1","source":"policy.md"}],"missing":""}'
    )


def test_workflow_e2e() -> None:
    cfg = AppConfig(max_iterations=2)
    nodes = GraphNodes(
        config=cfg,
        llm_client=LLMClient(backend, retries=0),
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        logger=FakeLogger(),
    )
    service = RAGService(app_config=cfg, nodes=nodes)
    result = service.run("Сколько дней отпуска по политике компании?")
    assert result.status == "ok"
    assert "10" in result.answer
    assert result.needs_retrieval is True
    assert result.latency_ms["total"] >= 0.0
