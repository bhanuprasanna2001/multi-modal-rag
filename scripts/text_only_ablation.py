#!/usr/bin/env python3
import os, sys, json

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmrag.index import build_index


def main():
    # Temporarily ignore captions by renaming file if present
    art = os.path.join(REPO_ROOT, "artifacts")
    caps = os.path.join(art, "captions.jsonl")
    tmp = os.path.join(art, "captions.disabled.jsonl")
    if os.path.exists(caps):
        os.replace(caps, tmp)
        print("Temporarily disabled captions for ablation")
    try:
        n_items, dim = build_index()
        print(json.dumps({"index_items": n_items, "dim": dim}))
    finally:
        if os.path.exists(tmp):
            os.replace(tmp, caps)
            print("Restored captions file")


if __name__ == "__main__":
    main()
