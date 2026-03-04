#!/usr/bin/env python3
"""Ablation study: rebuild index without image captions to measure their impact."""

import json
import os

from mmrag.index import build_index
from mmrag.utils import Paths


def main():
    paths = Paths()
    captions = paths.captions_jsonl
    backup = captions + ".disabled"

    if os.path.exists(captions):
        os.replace(captions, backup)
        print("Disabled captions for ablation")

    try:
        stats = build_index()
        print(json.dumps(stats, indent=2))
    finally:
        if os.path.exists(backup):
            os.replace(backup, captions)
            print("Restored captions")


if __name__ == "__main__":
    main()
