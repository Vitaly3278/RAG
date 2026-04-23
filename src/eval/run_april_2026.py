from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset, load_dataset
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness

from src.bootstrap import build_rag_service
from src.config import load_config


def build_eval_rows(limit: int = 25) -> list[dict]:
    cfg = load_config()
    service = build_rag_service(cfg)
    ds = load_dataset("squad", split=f"validation[:{limit}]")

    rows: list[dict] = []
    for idx, row in enumerate(ds):
        question = str(row.get("question", "")).strip()
        ground_truths = row.get("answers", {}).get("text", []) or []
        ground_truth = str(ground_truths[0]).strip() if ground_truths else ""
        if not question or not ground_truth:
            continue

        result = service.run(question, request_id=f"eval-{idx}")
        contexts = [str(c.get("text", "")).strip() for c in result.reranked_chunks if c.get("text")]
        rows.append(
            {
                "question": question,
                "contexts": contexts,
                "answer": result.answer,
                "ground_truth": ground_truth,
            }
        )
    return rows


def run_eval(limit: int, output_path: str, judge_model: str) -> None:
    rows = build_eval_rows(limit=limit)
    ds = Dataset.from_list(rows)
    cfg = load_config()
    ollama_base = os.getenv("RAG_LLM_OLLAMA_URL", cfg.llm.ollama_url)
    ragas_llm = ChatOllama(
        model=judge_model,
        base_url=ollama_base,
        temperature=0.0,
    )
    ragas_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=ollama_base,
    )
    ragas_llm_wrapper = LangchainLLMWrapper(ragas_llm)
    ragas_embedding_wrapper = LangchainEmbeddingsWrapper(ragas_embeddings)
    metrics = [
        Faithfulness(llm=ragas_llm_wrapper),
        ResponseRelevancy(llm=ragas_llm_wrapper, embeddings=ragas_embedding_wrapper),
        ContextPrecision(llm=ragas_llm_wrapper),
    ]
    scores = evaluate(
        ds,
        metrics=metrics,
        llm=ragas_llm_wrapper,
        embeddings=ragas_embedding_wrapper,
    )
    df = scores.to_pandas()

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_count": len(rows),
        "metrics_mean": {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
        },
        "rows": json.loads(df.to_json(orient="records")),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run April 2026 RAGAS eval on public test questions.")
    parser.add_argument("--limit", type=int, default=25, help="Question count (20-30 recommended).")
    parser.add_argument("--output", default="eval_results/april_2026.json", help="Output JSON path.")
    parser.add_argument(
        "--judge-model",
        default="qwen2.5-coder:1.5b",
        help="Ollama model for RAGAS judge metrics.",
    )
    args = parser.parse_args()
    run_eval(limit=args.limit, output_path=args.output, judge_model=args.judge_model)
    print(f"Saved RAGAS report to {args.output}")


if __name__ == "__main__":
    main()
