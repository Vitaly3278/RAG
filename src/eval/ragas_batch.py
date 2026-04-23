from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness


def run_batch_eval(input_path: str, output_path: str) -> None:
    rows = json.loads(Path(input_path).read_text(encoding="utf-8"))
    ds = Dataset.from_list(rows)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    Path(output_path).write_text(result.to_pandas().to_json(orient="records"), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline RAGAS evaluation batch.")
    parser.add_argument("--input", required=True, help="Path to JSON list with question/contexts/answer/ground_truth.")
    parser.add_argument("--output", default="ragas_results.json", help="Where to write evaluation output.")
    args = parser.parse_args()
    run_batch_eval(args.input, args.output)
