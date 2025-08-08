#!/usr/bin/env python3
import os, sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmrag.rag import answer_question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="Question to ask")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    out = answer_question(args.question, k=args.k)
    print("Answer:\n", out["answer"])  # contains citations
    print("\nTop hits:")
    for h in out["hits"]:
        tag = f"({h.get('doc_id')}:{h.get('page')})"
        print(f"- {h['type']} {tag} score={h['score']:.3f}")


if __name__ == "__main__":
    main()
