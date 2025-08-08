#!/usr/bin/env python3
import os, sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmrag.rag import answer_question


def main():
    qs = [
        "What is the operating temperature range?",
        "Which pin is labeled EN?",
        "What connector type is specified?",
    ]
    for q in qs:
        out = answer_question(q)
        print("Q:", q)
        print("A:", out["answer"])
        print("Top hit types:", [h["type"] for h in out["hits"]])
        print("---")


if __name__ == "__main__":
    main()
