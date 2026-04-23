from __future__ import annotations

from src.llm.client import LLMClient


def test_router_json_with_fence_and_retry() -> None:
    calls: list[str] = []

    def backend(prompt: str, **kwargs):  # noqa: ANN003
        calls.append(prompt)
        if len(calls) == 1:
            return "```json\n{invalid}\n```"
        return '```json\n{"needs_retrieval": true, "rewritten_query": "vacation policy"}\n```'

    client = LLMClient(backend, retries=2)
    payload = client.invoke_json("router")
    assert payload["needs_retrieval"] is True
    assert payload["rewritten_query"] == "vacation policy"
    assert len(calls) == 2
    assert "Previous output was invalid JSON." in calls[1]


def test_self_correction_json_with_extra_text() -> None:
    def backend(prompt: str, **kwargs):  # noqa: ANN003
        return (
            "analysis...\n"
            '{"is_faithful": true, "covers_query": false, "hallucinated_fragments": ["x"], '
            '"needs_correction": true, "correction_query": "policy pto 2026"}\n'
            "done"
        )

    client = LLMClient(backend, retries=0)
    payload = client.invoke_json("self-correction")
    assert payload["is_faithful"] is True
    assert payload["needs_correction"] is True
    assert payload["correction_query"] == "policy pto 2026"
