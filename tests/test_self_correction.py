from __future__ import annotations

from src.config import AppConfig
from src.graph.nodes import GraphNodes
from src.graph.state import GraphState
from src.llm.client import LLMClient


class FakeRetriever:
    def search(self, query: str, top_k: int = 5):  # noqa: ANN001
        return [{"id": "1", "text": "doc", "source": "kb", "score": 0.9}]


class FakeReranker:
    def rerank(self, query: str, chunks: list[dict], top_k: int = 5):  # noqa: ANN001
        return chunks[:top_k]


class FakeLogger:
    def info(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


def test_self_correction_increments_iteration() -> None:
    def backend(prompt: str, **kwargs):  # noqa: ANN003
        if "аудитор" in prompt.lower():
            return (
                '{"is_faithful": false, "covers_query": false, '
                '"hallucinated_fragments":["x"], "needs_correction": true, '
                '"correction_query":"refined query"}'
            )
        return '{"status":"ok","answer":"a","citations":[],"missing":""}'

    nodes = GraphNodes(
        config=AppConfig(max_iterations=2),
        llm_client=LLMClient(backend, retries=0),
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        logger=FakeLogger(),
    )
    state = GraphState(query="q", answer="bad", reranked_chunks=[{"text": "ctx"}]).to_dict()
    out = GraphState.model_validate(nodes.self_correction(state))
    assert out.needs_correction is True
    assert out.iteration == 1
    assert out.query == "refined query"


def test_self_correction_stops_at_max_iterations() -> None:
    def backend(prompt: str, **kwargs):  # noqa: ANN003
        if "аудитор" in prompt.lower():
            return (
                '{"is_faithful": false, "covers_query": false, '
                '"hallucinated_fragments":["x"], "needs_correction": true, '
                '"correction_query":"retry"}'
            )
        return '{"status":"ok","answer":"a","citations":[],"missing":""}'

    nodes = GraphNodes(
        config=AppConfig(max_iterations=2),
        llm_client=LLMClient(backend, retries=0),
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        logger=FakeLogger(),
    )
    state = GraphState(query="q", answer="bad", iteration=2, reranked_chunks=[{"text": "ctx"}]).to_dict()
    out = GraphState.model_validate(nodes.self_correction(state))
    assert out.needs_correction is False
    assert out.status == "low_confidence"
