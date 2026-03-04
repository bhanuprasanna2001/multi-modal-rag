#!/usr/bin/env python3
"""Quick smoke test: ask a few questions and print results."""

from mmrag.rag import answer_question


def main():
    questions = [
        "What is the operating temperature range?",
        "Which pin is labeled EN?",
        "What connector type is specified?",
    ]
    for q in questions:
        result = answer_question(q)
        m = result["metrics"]
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {[h['type'] for h in result['hits']]}")
        print(f"Latency: {m['total_ms']:.0f}ms | Cost: ${m['cost_usd']:.5f}")
        print("---")


if __name__ == "__main__":
    main()
