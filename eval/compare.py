#!/usr/bin/env python3
"""A/B comparison: run the same questions under two configurations and compare."""

import argparse
import json
import os

from mmrag.rag import answer_question
from dotenv import load_dotenv

load_dotenv()


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_config(questions: list[dict], k: int, label: str) -> dict:
    """Run all questions with a given k and return aggregate stats."""
    correct = 0
    latencies: list[float] = []
    total_cost = 0.0

    for i, q in enumerate(questions, 1):
        result = answer_question(q["q"], k=k)
        m = result["metrics"]
        answer = result["answer"]
        latencies.append(m["total_ms"])
        total_cost += m["cost_usd"]

        has_citation = ":" in answer and ("(" in answer or "[" in answer)
        if "expected_contains" in q:
            ok = q["expected_contains"].lower() in answer.lower() and has_citation
        else:
            ok = has_citation
        correct += int(ok)

        status = "PASS" if ok else "FAIL"
        print(f"  [{label}] [{i}/{len(questions)}] {status} "
              f"({m['total_ms']:.0f}ms) {q['q'][:50]}")

    n = len(questions)
    sorted_lat = sorted(latencies)
    return {
        "label": label,
        "k": k,
        "questions": n,
        "correct": correct,
        "accuracy": round(correct / n, 3) if n else 0,
        "avg_latency_ms": round(sum(sorted_lat) / len(sorted_lat), 1) if sorted_lat else 0,
        "p50_latency_ms": round(sorted_lat[len(sorted_lat) // 2], 1) if sorted_lat else 0,
        "p95_latency_ms": round(sorted_lat[int(0.95 * len(sorted_lat))], 1) if sorted_lat else 0,
        "total_cost_usd": round(total_cost, 5),
        "avg_cost_usd": round(total_cost / n, 6) if n else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="A/B comparison")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--k-a", type=int, default=3, help="k for config A")
    parser.add_argument("--k-b", type=int, default=7, help="k for config B")
    parser.add_argument("--output", default="eval/comparison.json")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"Running A/B comparison: k={args.k_a} vs k={args.k_b} "
          f"on {len(questions)} questions\n")

    print(f"--- Config A (k={args.k_a}) ---")
    stats_a = run_config(questions, args.k_a, f"k={args.k_a}")

    print(f"\n--- Config B (k={args.k_b}) ---")
    stats_b = run_config(questions, args.k_b, f"k={args.k_b}")

    report = {"config_a": stats_a, "config_b": stats_b}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'A (k=' + str(args.k_a) + ')':>15} {'B (k=' + str(args.k_b) + ')':>15}")
    print("-" * 60)
    for key in ["accuracy", "avg_latency_ms", "p50_latency_ms",
                "p95_latency_ms", "total_cost_usd", "avg_cost_usd"]:
        va = stats_a[key]
        vb = stats_b[key]
        fmt = ".5f" if "cost" in key else (".3f" if "accuracy" in key else ".1f")
        print(f"{key:<25} {va:>{15}{fmt}} {vb:>{15}{fmt}}")
    print("=" * 60)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
