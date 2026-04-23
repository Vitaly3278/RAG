Ты — senior LLM engineer. Сгенерируй полный production-ready код RAG-системы с self-correction на Python.

## 0) Жесткие ограничения генерации
- Используй только: LangGraph, Qdrant, BGE-reranker, Qwen3-8B-Instruct, RAGAS, OpenTelemetry + Phoenix, FastAPI, pytest.
- Не предлагай альтернативные фреймворки.
- Выводи код строго "по файлам" (см. раздел 2).
- Сначала выведи `architecture.md`, затем все остальные файлы.
- Все контракты (state/API/JSON) соблюдать строго, без "примерно".

## 1) Цель системы
Собрать RAG-пайплайн:
`Router -> Retriever -> Reranker -> Generator -> SelfCorrection -> Output`

Требования:
1. Адаптивный retrieval: роутер решает, нужен ли поиск, и переписывает запрос.
2. Самокоррекция: при неfaithful ответе запускать повторный поиск с correction_query.
3. Максимум 2 итерации коррекции, затем graceful fallback.
4. Offline-оценка quality через RAGAS (faithfulness, answer_relevancy, context_precision).
5. Продакшен-аспекты: JSON-логирование, метрики latency/throughput, трассировка, отказоустойчивость.

## 2) Формат ответа "по файлам" (обязателен)
Для каждого файла используй только этот шаблон:

FILE: path/to/file.ext
```language
<полное содержимое файла>
```

Порядок:
1) `architecture.md`
2) код в `src/`
3) промпты в `prompts/`
4) конфиги в `configs/`
5) тесты в `tests/`
6) `docker-compose.yml`
7) `Dockerfile`
8) `README.md`

## 3) Целевая структура проекта
- `src/graph/state.py`
- `src/graph/nodes.py`
- `src/graph/workflow.py`
- `src/retrieval/qdrant_client.py`
- `src/retrieval/reranker.py`
- `src/llm/client.py`
- `src/api/main.py`
- `src/observability/logging.py`
- `src/observability/tracing.py`
- `src/eval/ragas_batch.py`
- `src/config.py`
- `prompts/router.txt`
- `prompts/generator.txt`
- `prompts/self_correction.txt`
- `prompts/evaluator.txt`
- `configs/settings.yaml`
- `tests/test_workflow.py`
- `tests/test_router_schema.py`
- `tests/test_self_correction.py`
- `tests/test_api.py`
- `docker-compose.yml`
- `Dockerfile`
- `README.md`

## 4) Контракт состояния графа (обязательный)
Определи `GraphState` (Pydantic), минимум поля:
- `request_id: str`
- `query: str`
- `needs_retrieval: bool`
- `rewritten_query: str`
- `retrieved_chunks: list[dict]`
- `reranked_chunks: list[dict]`
- `answer: str`
- `status: str`  # ok | insufficient_context | low_confidence | fallback
- `hallucinated_fragments: list[str]`
- `covers_query: bool`
- `is_faithful: bool`
- `needs_correction: bool`
- `correction_query: str`
- `iteration: int`
- `max_iterations: int`  # default=2
- `latency_ms: dict[str, float]`
- `errors: list[str]`

## 5) Контракт API (обязательный)
FastAPI endpoints:
1. `POST /ask`
   - Request:
     ```json
     { "query": "string", "request_id": "optional-string" }
     ```
   - Response (строго):
     ```json
     {
       "request_id": "string",
       "status": "ok|insufficient_context|low_confidence|fallback",
       "answer": "string",
       "citations": [{"id":"string","source":"string","score":0.0}],
       "iteration_count": 0,
       "needs_retrieval": true,
       "timings_ms": {"router":0.0,"retriever":0.0,"reranker":0.0,"generator":0.0,"self_correction":0.0,"total":0.0}
     }
     ```
2. `GET /health` -> `{ "status": "ok" }`
3. `GET /metrics` -> Prometheus-compatible текст.

## 6) Логика переходов LangGraph (обязательная)
- Старт в `Router`.
- Если `needs_retrieval == false` -> `Output` (базовый/fallback ответ без retriever).
- Если `true` -> `Retriever` -> `Reranker` -> `Generator` -> `SelfCorrection`.
- В `SelfCorrection`:
  - если `needs_correction == true` и `iteration < max_iterations`: увеличить `iteration`, обновить `query=correction_query`, перейти к `Retriever`.
  - иначе завершить в `Output`.
- Если Qdrant/LLM недоступен: не падать, вернуть `status="fallback"` и безопасный ответ.

## 7) Контракты промптов (строгий JSON)

### 7.1 Router prompt (`prompts/router.txt`)
Роль: решить, нужен ли retrieval.
Выход строго JSON:
```json
{
  "needs_retrieval": true,
  "rewritten_query": "string"
}
```
Правила:
- Факты/цифры/ссылки/политики -> `needs_retrieval=true`.
- Общие/чатовые/форматные запросы -> `needs_retrieval=false`.
- Только валидный JSON без дополнительного текста.

### 7.2 Generator prompt (`prompts/generator.txt`)
Роль: ответ только по контексту.
Выход строго JSON:
```json
{
  "status": "ok|insufficient_context",
  "answer": "string",
  "citations": [{"id":"string","source":"string"}],
  "missing": "string"
}
```
Правила:
- Никаких выдуманных фактов.
- Если контекста недостаточно -> `status="insufficient_context"` и заполнить `missing`.

### 7.3 Self-correction prompt (`prompts/self_correction.txt`)
Выход строго JSON:
```json
{
  "is_faithful": true,
  "covers_query": true,
  "hallucinated_fragments": [],
  "needs_correction": false,
  "correction_query": ""
}
```

### 7.4 Evaluator prompt (`prompts/evaluator.txt`)
Выход строго JSON:
```json
{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "context_precision": 0.0,
  "reasoning": "string"
}
```

## 8) Продакшен-параметры
- Температуры:
  - router/self-correction/evaluator: `0.0-0.2`
  - generator: `0.3-0.4`
- Таймауты:
  - LLM node timeout: `5s`
  - Qdrant timeout: `2s`
- Retry:
  - до `2` попыток только на JSON-валидацию ответа LLM.
- Кэш:
  - ключ `hash(query + top_k_context_ids)`, TTL `24-72h`.
- Логи:
  - JSON logs с `request_id`, `node`, `latency_ms`, `tokens_in`, `tokens_out`, `status`.

## 9) Тестовый минимум (обязательный)
- `tests/test_router_schema.py`: валидация JSON-схемы роутера.
- `tests/test_self_correction.py`: проверка ветвления correction/retry max=2.
- `tests/test_workflow.py`: e2e пайплайн с mock Qdrant и mock LLM.
- `tests/test_api.py`: `POST /ask`, `GET /health`, `GET /metrics`.

## 10) Критерии готовности (Definition of Done)
1. `pytest` проходит.
2. Пайплайн не падает при недоступном Qdrant, отдает `fallback`.
3. В трейсах Phoenix видны все узлы и задержки.
4. Есть batched/offline RAGAS-скрипт.
5. README содержит:
   - схему архитектуры,
   - как запускать локально (`docker compose up`),
   - как запускать тесты,
   - целевые KPI: `faithfulness > 0.85`, `latency p95 < 2.5s`.

## 11) Что вернуть прямо сейчас
Верни полный код проекта по структуре из раздела 3 и формату из раздела 2.
Ничего не пропускай, без заглушек вида TODO.
