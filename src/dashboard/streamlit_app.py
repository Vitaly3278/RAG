from __future__ import annotations

import json
from dataclasses import dataclass

import requests
import streamlit as st


@dataclass
class ApiConfig:
    base_url: str
    timeout_seconds: float


def _default_config() -> ApiConfig:
    return ApiConfig(base_url="http://localhost:8000", timeout_seconds=30.0)


def _parse_prometheus_subset(raw: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("rag_requests_total"):
            metrics["rag_requests_total"] = float(stripped.split()[-1])
        elif stripped.startswith("rag_request_latency_seconds_sum"):
            metrics["rag_request_latency_seconds_sum"] = float(stripped.split()[-1])
        elif stripped.startswith("rag_request_latency_seconds_count"):
            metrics["rag_request_latency_seconds_count"] = float(stripped.split()[-1])
    return metrics


def render() -> None:
    st.set_page_config(page_title="ContextGuard Dashboard", layout="wide")
    st.title("ContextGuard RAG Dashboard")

    cfg = _default_config()
    st.sidebar.header("API Settings")
    cfg.base_url = st.sidebar.text_input("Base URL", value=cfg.base_url).rstrip("/")
    cfg.timeout_seconds = st.sidebar.slider("Timeout (seconds)", 5.0, 120.0, 30.0, step=5.0)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Question",
            value="Какая политика отпусков в компании?",
            height=120,
        )
    with col2:
        send = st.button("Ask", type="primary", use_container_width=True)

    if send:
        try:
            response = requests.post(
                f"{cfg.base_url}/ask",
                json={"query": query},
                timeout=cfg.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
            return

        st.subheader("Answer")
        st.write(payload.get("answer", ""))
        st.caption(f"status={payload.get('status')} request_id={payload.get('request_id')}")

        citations = payload.get("citations") or []
        st.subheader("Sources")
        if citations:
            st.dataframe(citations, use_container_width=True)
        else:
            st.info("No citations returned.")

        st.subheader("Node Metrics")
        timings = payload.get("timings_ms") or {}
        if timings:
            rows = [{"node": k, "latency_ms": float(v)} for k, v in timings.items()]
            st.bar_chart(data=rows, x="node", y="latency_ms")
            st.code(json.dumps(timings, ensure_ascii=False, indent=2), language="json")
        else:
            st.info("No timings in response.")

    st.divider()
    st.subheader("Prometheus Snapshot")
    if st.button("Refresh /metrics"):
        try:
            metrics_response = requests.get(f"{cfg.base_url}/metrics", timeout=cfg.timeout_seconds)
            metrics_response.raise_for_status()
            parsed = _parse_prometheus_subset(metrics_response.text)
            if parsed:
                st.json(parsed)
            else:
                st.warning("No expected rag_* metrics found.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Metrics fetch failed: {exc}")


if __name__ == "__main__":
    render()
