#!/usr/bin/env python3
import os, sys
import json
import time
from typing import List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmrag.rag import answer_question


def load_questions(path: str) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line))
    return qs


def contains_match(ans: str, expected: str) -> bool:
    return expected.lower() in ans.lower()


def main():
    qs = load_questions("eval/questions.jsonl")
    n = len(qs)
    correct = 0
    correct_img = 0
    img_count = 0

    t0 = time.time()
    latencies: List[float] = []
    for q in qs:
        s0 = time.time()
        out = answer_question(q["q"])  # includes citations
        dt = time.time() - s0
        latencies.append(dt)

        ans = out["answer"]
        ok = False
        if "expected_contains" in q:
            ok = contains_match(ans, q["expected_contains"]) and "(" in ans and ")" in ans
        else:
            ok = "(" in ans and ")" in ans
        correct += int(ok)

        if q.get("image_centric"):
            img_count += 1
            # success if any caption appears in hits
            img_hit = any(h.get("type") == "caption" for h in out["hits"]) and ok
            correct_img += int(img_hit)

    total_time = time.time() - t0
    acc = correct / n if n else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0

    print(json.dumps({
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "image_qs": img_count,
        "image_correct": correct_img,
        "avg_latency_sec": avg_lat,
        "p95_latency_sec": p95_lat,
        "total_time_sec": total_time,
    }, indent=2))


if __name__ == "__main__":
    main()
