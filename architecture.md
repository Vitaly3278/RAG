# Production RAG Architecture

## Goal

Build a production-ready RAG service with adaptive retrieval, self-correction, offline quality evaluation, and observability.

## State Graph

`Router -> Retriever -> Reranker -> Generator -> SelfCorrection -> Output`

Routing logic:

- If `needs_retrieval == false`: go directly to `Output`.
- If `needs_retrieval == true`: run retrieval path.
- If self-correction says retry is needed and `iteration < max_iterations`: loop back to `Retriever` with `correction_query`.
- If retries are exhausted or correction not needed: finish at `Output`.

## Node Contracts

- **Router**: returns `needs_retrieval`, `rewritten_query`.
- **Retriever**: returns `retrieved_chunks`.
- **Reranker**: returns `reranked_chunks`.
- **Generator**: returns `status`, `answer`, `citations`.
- **SelfCorrection**: returns faithfulness/completeness flags and correction controls.
- **Output**: normalizes final response and timings.

## Reliability

- Every node is wrapped with timeout/error-safe behavior.
- On Qdrant/LLM failures, graph returns `status=fallback` instead of crashing.
- Max correction loops: `2`.

## Observability

- JSON logging per node with `request_id`, `latency_ms`, token usage placeholders, and status.
- OpenTelemetry tracing with Phoenix endpoint support.
- Prometheus metrics endpoint for throughput/latency counters and histograms.

## API

- `POST /ask` main inference endpoint.
- `GET /health` health status.
- `GET /metrics` Prometheus metrics.

## Evaluation

- Offline batch script computes:
  - faithfulness
  - answer_relevancy
  - context_precision
- Uses RAGAS for periodic quality reports.

## Testing Strategy

- Router schema validation.
- Self-correction branching and retry cap.
- End-to-end graph execution with mocked Qdrant and mocked LLM.
- API contract tests for all endpoints.