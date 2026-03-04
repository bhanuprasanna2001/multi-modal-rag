#!/usr/bin/env python3
"""RAGAS evaluation: faithfulness, answer relevancy, context precision."""

import argparse
import json
import os
import sys
import warnings

# Fix macOS OMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv

load_dotenv()

# Suppress RAGAS deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument(
        "--questions",
        default="eval/questions.jsonl",
        help="Path to questions JSONL file",
    )
    parser.add_argument(
        "--output",
        default="eval/ragas_results.json",
        help="Path to write results JSON",
    )
    args = parser.parse_args()

    # Late imports so missing deps produce a clear error
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("Install RAGAS first:  pip install ragas datasets langchain-openai")
        sys.exit(1)

    from mmrag.rag import answer_question

    # Load questions
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    # RAGAS 0.4+ uses: user_input, response, retrieved_contexts, reference
    rows: dict[str, list] = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    total_cost = 0.0
    for i, q in enumerate(questions, 1):
        result = answer_question(q["q"])
        total_cost += result["metrics"]["cost_usd"]

        rows["user_input"].append(q["q"])
        rows["response"].append(result["answer"])
        rows["retrieved_contexts"].append([h["content"] for h in result["hits"]])
        rows["reference"].append(q.get("expected", ""))

        print(f"[{i}/{len(questions)}] {q['q'][:60]}  "
              f"(${result['metrics']['cost_usd']:.5f})")

    ds = Dataset.from_dict(rows)

    metrics = [faithfulness, answer_relevancy, context_precision]
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    result = evaluate(ds, metrics=metrics, embeddings=embeddings)

    def _scalar(v):
        """RAGAS 0.4+ may return a list per metric; average if so."""
        if isinstance(v, list):
            nums = [x for x in v if isinstance(x, (int, float))]
            return round(sum(nums) / len(nums), 4) if nums else 0.0
        return round(v, 4)

    report = {
        "faithfulness": _scalar(result["faithfulness"]),
        "answer_relevancy": _scalar(result["answer_relevancy"]),
        "context_precision": _scalar(result["context_precision"]),
        "questions_evaluated": len(questions),
        "rag_cost_usd": round(total_cost, 5),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{json.dumps(report, indent=2)}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
