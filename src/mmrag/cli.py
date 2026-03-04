"""CLI entry points for the multi-modal RAG pipeline."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from loguru import logger


def _setup():
    """Load .env and quiet library logs so only progress bars + results show."""
    load_dotenv()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def build():
    """Run the full pipeline: ingest PDFs → caption images → build index."""
    _setup()

    from .ingest import ingest_pdfs
    from .caption import generate_captions
    from .index import build_index

    print("[1/3] Ingesting PDFs...")
    stats = ingest_pdfs()
    print(f"  → {stats['chunks']} text chunks, {stats['images']} images "
          f"from {stats['pdfs']} PDFs\n")

    print("[2/3] Captioning images...")
    n_captions = generate_captions()
    print(f"  → {n_captions} captions generated\n")

    print("[3/3] Building search index...")
    n_items, dim = build_index()
    print(f"  → {n_items} items indexed ({dim}-dim embeddings)\n")

    print("Done.")


def query():
    """Ask a question from the command line."""
    _setup()

    from .rag import answer_question

    parser = argparse.ArgumentParser(description="Ask a question about the datasheets")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    result = answer_question(args.question, k=args.k)

    print(f"\nAnswer: {result['answer']}\n")

    m = result["metrics"]
    print(f"Metrics: {m['total_ms']:.0f}ms total | "
          f"{m['retrieval_ms']:.0f}ms retrieval | "
          f"{m['generation_ms']:.0f}ms generation | "
          f"${m['cost_usd']:.5f}")

    print("\nSources:")
    for hit in result["hits"]:
        tag = f"({hit.get('doc_id')}:{hit.get('page')})"
        print(f"  {hit['type']} {tag} score={hit['score']:.4f}")
