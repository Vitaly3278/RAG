# ContextGuard RAG (LangGraph + Qdrant)

Production-ready каркас RAG-системы с:
- адаптивным retrieval,
- self-correction циклом,
- FastAPI API,
- трассировкой и метриками,
- offline оценкой качества через RAGAS.

## Что реализовано

Граф выполнения:
`Router -> Retriever -> Reranker -> Generator -> SelfCorrection -> Output`

Поведение пайплайна:
1. `Router` решает, нужен ли retrieval, и переписывает запрос.
2. `Retriever` получает чанки из Qdrant.
3. `Reranker` сортирует контекст (интерфейс под BGE-reranker).
4. `Generator` формирует ответ строго по контексту.
5. `SelfCorrection` проверяет faithful/completeness и, при необходимости, запускает повторный поиск.
6. `Output` возвращает нормализованный ответ и тайминги.

Если Qdrant/LLM недоступен, система не падает и возвращает `fallback`.

## Структура проекта

- `architecture.md` — описание архитектуры и потоков.
- `src/graph/` — состояние, узлы и сборка LangGraph.
- `src/retrieval/` — клиент Qdrant и reranker.
- `src/llm/` — JSON-ориентированный LLM-клиент.
- `src/api/` — FastAPI приложение.
- `src/observability/` — JSON-логирование и OTEL tracing.
- `src/eval/` — offline batch-оценка RAGAS.
- `prompts/` — контракты промптов.
- `configs/` — runtime-настройки.
- `tests/` — набор pytest-тестов.

## Быстрый старт (локально)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

Сервис поднимется на `http://127.0.0.1:8000`.

## Запуск через Docker

```bash
docker compose up --build
```

По умолчанию поднимаются:
- API на `:8000`
- Qdrant на `:6333`

## API контракт

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

### `GET /health`

```json
{"status":"ok"}
```

### `GET /metrics`

Prometheus-совместимые метрики.

## Тесты

```bash
pytest -q
```

Покрываются:
- валидация схемы роутера,
- ветвление self-correction и лимит итераций,
- e2e граф с mock Qdrant/LLM,
- API контракт и метрики.

## Offline оценка качества (RAGAS)

Подготовь JSON-датасет и запусти:

```bash
python -m src.eval.ragas_batch --input data/eval.json --output ragas_results.json
```

Целевые метрики:
- `faithfulness > 0.85`
- `latency p95 < 2.5s`

## Наблюдаемость

- JSON-логи по узлам (`request_id`, `node`, `latency_ms`, `status`, токены).
- OpenTelemetry интеграция в `src/observability/tracing.py`.
- Метрики для Prometheus через `GET /metrics`.

## Конфигурация

Основные параметры в `configs/settings.yaml`:
- `max_iterations` — максимум циклов коррекции.
- `qdrant.*` — endpoint/collection/top_k/timeout.
- `llm.*` — модель, температуры и retry JSON-парсинга.
- `fallback_answer` — безопасный ответ при деградации.

## Что важно перед продом

- Подключить реальный LLM backend для `Qwen3-8B-Instruct`.
- Добавить ingestion-пайплайн документов в Qdrant.
- Настроить экспортер трейсинга в Phoenix.
- Запустить нагрузочные тесты и проверить p95 latency.
