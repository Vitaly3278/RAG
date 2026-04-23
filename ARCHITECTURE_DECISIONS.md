# Architecture Decisions

## 1) Why LangGraph

**Decision:** use LangGraph as the orchestration layer for RAG.

**Why:**
- Native state-machine model fits router/retriever/generator/self-correction loops.
- Explicit graph edges make retry/correction behavior deterministic and testable.
- Easy to isolate node-level failures and return fallback instead of hard crash.

**Trade-offs:**
- More boilerplate than a single linear chain.
- Team needs to understand graph/state semantics.

## 2) Why Qdrant

**Decision:** use Qdrant as vector database for retrieval.

**Why:**
- Fast local/dev startup and simple Docker deployment.
- Good filtering + vector search API and predictable operational model.
- Compatible with lightweight ingestion scripts and custom embedding pipeline.

**Trade-offs:**
- Additional service to operate in production.
- Requires schema/index lifecycle management and backup strategy.

## 3) Why JSON-oriented LLM contract

**Decision:** force router/generator/self-correction responses to strict JSON.

**Why:**
- Stable machine-to-machine contract between graph nodes.
- Easy validation and safe fallbacks if parsing fails.
- Enables self-correction loop without fragile text parsing heuristics.

**Trade-offs:**
- Prompting must be stricter and can be slower.
- Some models still require retries and correction prompts.

## 4) Why offline RAGAS batch

**Decision:** keep offline batch evaluation alongside online telemetry.

**Why:**
- Gives repeatable quality snapshots for regressions and prompt/model changes.
- Separates runtime performance monitoring from semantic quality assessment.

**Trade-offs:**
- Evaluation is expensive/time-consuming on local models.
- Scores can drift with judge model changes.

## 5) Why reverse proxy in production compose

**Decision:** expose only reverse proxy to external traffic.

**Why:**
- Central place for TLS/rate-limit/auth controls (future-ready).
- Hides internal service topology from clients.
- Simplifies blue-green style deployment patterns.

**Trade-offs:**
- One more component to monitor and configure.
- Misconfiguration can become a single point of failure.
