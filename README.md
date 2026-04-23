# ContextGuard RAG (LangGraph + Qdrant + Ollama)

Production-ready каркас RAG-системы с адаптивным retrieval, self-correction контуром, FastAPI API, визуализацией, мониторингом и offline-оценкой качества.

## Что внутри

- Граф: `Router -> Retriever -> Reranker -> Generator -> SelfCorrection -> Output`
- Роутер решает, нужен ли retrieval, и переписывает запрос.
- Генератор и self-correction работают по строгому JSON-контракту.
- Деградация безопасная: при проблемах с LLM/Qdrant отдается `fallback`.
- Наблюдаемость: `GET /metrics`, health/live/ready endpoints.

## Структура проекта

- `src/api/` — FastAPI приложение.
- `src/graph/` — state + узлы + workflow (LangGraph).
- `src/llm/` — JSON-клиент и Ollama backend.
- `src/retrieval/` — Qdrant retriever, embeddings, reranker.
- `src/ingest/` — ingestion скрипты.
- `src/eval/` — RAGAS batch/апрельский прогон.
- `src/dashboard/` — Streamlit dashboard.
- `configs/` — runtime, monitoring, grafana/nginx provisioning.
- `tests/` — pytest покрытие.

## Быстрый старт (локально)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

API будет доступен на [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Docker Compose (dev stack)

```bash
docker compose up --build -d
```

Поднимаются сервисы:

- API: [http://localhost:8000](http://localhost:8000)
- Qdrant: `localhost:6333`
- Ollama: `localhost:11434`
- Streamlit: [http://localhost:8501](http://localhost:8501)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000)

После старта подгрузи модели в Ollama:

```bash
docker exec -it rag-ollama ollama pull qwen2.5:7b
# или
docker exec -it rag-ollama ollama pull qwen3:8b
docker exec -it rag-ollama ollama pull nomic-embed-text
```

## Визуализация

### Streamlit dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

Dashboard показывает:

- question -> answer -> status;
- список источников (`citations`);
- тайминги узлов (`router/retriever/reranker/generator/self_correction/total`);
- срез prometheus-метрик из `GET /metrics`.

### Скриншоты и демо

- Dashboard: `docs/screenshots/dashboard.png`
- Grafana: `docs/screenshots/grafana.png`
- Локальная демо-ссылка: [http://localhost:8501](http://localhost:8501)

Если скриншоты еще не добавлены, см. `docs/screenshots/README.md`.

## API

### `POST /ask`

Request:

```json
{
  "query": "string",
  "request_id": "optional-string"
}
```

Response:

```json
{
  "request_id": "string",
  "status": "ok|insufficient_context|low_confidence|fallback",
  "answer": "string",
  "citations": [{"id":"string","source":"string","score":0.0}],
  "iteration_count": 0,
  "needs_retrieval": true,
  "timings_ms": {
    "router": 0.0,
    "retriever": 0.0,
    "reranker": 0.0,
    "generator": 0.0,
    "self_correction": 0.0,
    "total": 0.0
  }
}
```

### Health/metrics endpoints

- `GET /health` — базовый статус.
- `GET /health/live` — liveness probe.
- `GET /health/ready` — readiness probe.
- `GET /metrics` — Prometheus-совместимые метрики.

## Мониторинг

В `docker-compose.yml` уже подключены:

- `prometheus` (scrape API по `api:8000/metrics`);
- `grafana` (datasource provisioning в `configs/grafana/provisioning`).

## Ingestion в Qdrant

Скрипт использует публичный `squad` и делает chunking:

```bash
python -m src.ingest.public_faq_to_qdrant --dataset squad --split "train[:120]" --limit 120 --chunk-size 120 --overlap 30 --recreate
```

## Offline evaluation (RAGAS)

```bash
python -m src.eval.ragas_batch --input data/eval.json --output ragas_results.json
python -m src.eval.run_april_2026 --limit 25 --output eval_results/april_2026.json --judge-model qwen2.5-coder:1.5b
```

Целевые ориентиры:

- `faithfulness > 0.85`
- `latency p95 < 2.5s`

### Таблица метрик (апрель 2026)

| Batch | Sample count | Faithfulness | Answer relevancy | Context precision |
|---|---:|---:|---:|---:|
| `eval_results/april_2026.json` | 20 | ~0.61 | 0.4726 | ~0.58 |

## Продовый деплой

Продовый compose с reverse proxy (nginx), health checks и graceful shutdown:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Внешняя точка входа: `http://localhost/` -> `nginx` -> API.

## Тесты

```bash
pytest -q
```

Покрываются схема роутера, ветвление self-correction, API-контракт, health/metrics endpoints и e2e workflow.

## Архитектурные решения

Подробные решения и trade-offs: `ARCHITECTURE_DECISIONS.md`.

