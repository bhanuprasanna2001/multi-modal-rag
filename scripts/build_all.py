#!/usr/bin/env python3
"""Build the complete RAG pipeline: ingest → caption → index."""

from mmrag.cli import build

if __name__ == "__main__":
    build()
