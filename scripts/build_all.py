#!/usr/bin/env python3
import os, sys

# Ensure repo_root/src is on sys.path for package imports
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmrag.ingest import ingest_pdfs
from mmrag.caption import generate_captions
from mmrag.index import build_index


def main():
    stats = ingest_pdfs()
    print("Ingestion:", stats)
    n_caps = generate_captions()
    print("Captioned:", n_caps)
    n_items, dim = build_index()
    print(f"Index built: {n_items} items, dim={dim}")


if __name__ == "__main__":
    main()
