from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_extended() -> None:
    client = TestClient(create_app())
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    assert live.status_code == 200
    assert ready.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.json()["status"] in {"ready", "not_ready"}


def test_ask_contract() -> None:
    client = TestClient(create_app())
    response = client.post("/ask", json={"query": "Какая политика отпусков?"})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "request_id",
        "status",
        "answer",
        "citations",
        "iteration_count",
        "needs_retrieval",
        "timings_ms",
    }


def test_metrics() -> None:
    client = TestClient(create_app())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_requests_total" in response.text
