#!/usr/bin/env python3
"""RAGAS evaluation: faithfulness, answer relevancy, context precision."""

import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()


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
    except ImportError:
        print("Install RAGAS first:  pip install ragas datasets")
        sys.exit(1)

    from mmrag.rag import answer_question

    # Load questions
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    rows: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    total_cost = 0.0
    for i, q in enumerate(questions, 1):
        result = answer_question(q["q"])
        total_cost += result["metrics"]["cost_usd"]

        rows["question"].append(q["q"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([h["content"] for h in result["hits"]])
        rows["ground_truth"].append(q.get("expected", ""))

        print(f"[{i}/{len(questions)}] {q['q'][:60]}  "
              f"(${result['metrics']['cost_usd']:.5f})")

    ds = Dataset.from_dict(rows)

    metrics = [faithfulness, answer_relevancy, context_precision]
    result = evaluate(ds, metrics=metrics)

    report = {
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
        "context_precision": round(result["context_precision"], 4),
        "questions_evaluated": len(questions),
        "rag_cost_usd": round(total_cost, 5),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{json.dumps(report, indent=2)}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
