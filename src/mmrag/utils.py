"""Shared utilities: paths, I/O helpers, text chunking, and timing."""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Iterable, Dict, Any

from loguru import logger


# ── Filesystem helpers ──


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def file_stem(path: str) -> str:
    """Return filename without extension: 'foo.pdf' → 'foo'."""
    return os.path.splitext(os.path.basename(path))[0]


def stable_hash(text: str) -> str:
    """Deterministic short hash for deduplication IDs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ── JSONL I/O ──


def save_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    """Write records as one-JSON-per-line to `path`."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {path}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ── Text chunking ──


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks, breaking at natural boundaries.

    Tries sentence boundaries first, then newlines, then spaces.
    Each chunk is at most `max_chars` characters. Consecutive chunks share
    `overlap` characters so information at boundaries isn't lost.
    """
    if len(text) <= max_chars:
        return [text]

    # Try progressively finer split points
    for pattern in [r"(?<=[.!?])\s+", r"\n+", r"\s+"]:
        segments = re.split(pattern, text)
        if len(segments) > 1:
            break
    else:
        # No split points at all — hard split by character
        segments = [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap)]
        return [s.strip() for s in segments if s.strip()]

    chunks: List[str] = []
    current = ""

    for segment in segments:
        if current and len(current) + len(segment) + 1 > max_chars:
            chunks.append(current.strip())
            # Carry over the tail of the previous chunk for overlap
            current = current[-overlap:] + " " + segment
        else:
            current = (current + " " + segment).strip() if current else segment

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


# ── Cost / latency tracking ──

# Pricing per 1M tokens (USD) — update if OpenAI changes prices
PRICING = {
    "text-embedding-3-small": {"input": 0.02},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class Metrics:
    """Accumulates cost and latency for a single request."""
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    embedding_tokens: int = 0
    generation_input_tokens: int = 0
    generation_output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "embedding_tokens": self.embedding_tokens,
            "generation_tokens": self.generation_input_tokens + self.generation_output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }

    def add_embedding_cost(self, model: str, tokens: int) -> None:
        self.embedding_tokens += tokens
        rate = PRICING.get(model, {}).get("input", 0)
        self.cost_usd += tokens * rate / 1_000_000

    def add_generation_cost(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.generation_input_tokens += input_tokens
        self.generation_output_tokens += output_tokens
        p = PRICING.get(model, {})
        self.cost_usd += input_tokens * p.get("input", 0) / 1_000_000
        self.cost_usd += output_tokens * p.get("output", 0) / 1_000_000


class Timer:
    """Simple context manager for timing blocks in milliseconds."""
    def __init__(self):
        self.ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self._start) * 1000


# ── Paths ──


@dataclass
class Paths:
    """Central registry of all file paths used by the pipeline."""

    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    @property
    def chunks_jsonl(self) -> str:
        return os.path.join(self.artifacts_dir, "chunks.jsonl")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.artifacts_dir, "images")

    @property
    def images_jsonl(self) -> str:
        return os.path.join(self.artifacts_dir, "images.jsonl")

    @property
    def captions_jsonl(self) -> str:
        return os.path.join(self.artifacts_dir, "captions.jsonl")

    @property
    def index_path(self) -> str:
        return os.path.join(self.artifacts_dir, "index.faiss")

    @property
    def bm25_path(self) -> str:
        return os.path.join(self.artifacts_dir, "bm25.pkl")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.artifacts_dir, "index_meta.json")
