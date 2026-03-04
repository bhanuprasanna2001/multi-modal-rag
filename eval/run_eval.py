#!/usr/bin/env python3
"""Evaluate RAG accuracy against a test question set."""

import json
import os

from mmrag.rag import answer_question
from dotenv import load_dotenv

load_dotenv()


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    questions = load_questions("eval/questions.jsonl")
    correct = 0
    image_total = 0
    image_correct = 0
    latencies: list[float] = []
    total_cost = 0.0

    for i, q in enumerate(questions, 1):
        result = answer_question(q["q"])
        m = result["metrics"]
        answer = result["answer"]
        latencies.append(m["total_ms"])
        total_cost += m["cost_usd"]

        # Check: answer contains expected text and has citations
        has_citation = ":" in answer and ("(" in answer or "[" in answer)
        if "expected_contains" in q:
            ok = q["expected_contains"].lower() in answer.lower() and has_citation
        else:
            ok = has_citation
        correct += int(ok)

        # Track image-centric questions separately
        if q.get("image_centric"):
            image_total += 1
            has_image_hit = any(h["type"] == "caption" for h in result["hits"])
            image_correct += int(ok and has_image_hit)

        status = "PASS" if ok else "FAIL"
        print(f"[{i}/{len(questions)}] {status} ({m['total_ms']:.0f}ms, "
              f"${m['cost_usd']:.5f}) {q['q'][:60]}")

    n = len(questions)
    sorted_lat = sorted(latencies)
    report = {
        "questions": n,
        "correct": correct,
        "accuracy": round(correct / n, 3) if n else 0,
        "image_questions": image_total,
        "image_correct": image_correct,
        "avg_latency_ms": round(sum(sorted_lat) / len(sorted_lat), 1) if sorted_lat else 0,
        "p50_latency_ms": round(sorted_lat[len(sorted_lat) // 2], 1) if sorted_lat else 0,
        "p95_latency_ms": round(sorted_lat[int(0.95 * len(sorted_lat))], 1) if sorted_lat else 0,
        "total_cost_usd": round(total_cost, 5),
        "avg_cost_usd": round(total_cost / n, 6) if n else 0,
    }
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
