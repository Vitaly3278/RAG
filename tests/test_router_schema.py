from __future__ import annotations

from pydantic import BaseModel


class RouterOutput(BaseModel):
    needs_retrieval: bool
    rewritten_query: str


def test_router_schema_validation_ok() -> None:
    payload = {"needs_retrieval": True, "rewritten_query": "kpi policy finance"}
    parsed = RouterOutput.model_validate(payload)
    assert parsed.needs_retrieval is True
    assert parsed.rewritten_query
