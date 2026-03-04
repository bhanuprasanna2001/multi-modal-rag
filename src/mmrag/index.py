"""Hybrid search index: OpenAI embeddings (dense) + BM25 (sparse) with rank fusion."""

from __future__ import annotations

import os
import json
import pickle
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm
import numpy as np
import faiss
from openai import OpenAI
from rank_bm25 import BM25Okapi

from .utils import Paths, read_jsonl, Metrics

EMBED_MODEL = "text-embedding-3-small"  # 1536-dim, $0.02/1M tokens
EMBED_BATCH_SIZE = 512  # OpenAI supports up to 2048 per call


# ── OpenAI Embedding helpers ──


def _embed_batch(client: OpenAI, texts: List[str], model: str) -> Tuple[List[List[float]], int]:
    """Embed a single batch. Returns (vectors, total_tokens)."""
    response = client.embeddings.create(input=texts, model=model)
    vectors = [d.embedding for d in response.data]
    tokens = response.usage.total_tokens
    return vectors, tokens


def embed_texts(
    texts: List[str], model: str = EMBED_MODEL, workers: int = 4
) -> Tuple[np.ndarray, int]:
    """Embed texts using OpenAI API with parallel batching.

    Returns (numpy array of vectors, total tokens used).
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for embeddings.")

    client = OpenAI()
    batches = [texts[i:i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]

    all_vectors: List[List[float]] = [[] for _ in texts]
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_embed_batch, client, batch, model): batch_idx
            for batch_idx, batch in enumerate(batches)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Embedding batches"
        ):
            batch_idx = futures[future]
            vectors, tokens = future.result()
            start = batch_idx * EMBED_BATCH_SIZE
            for j, vec in enumerate(vectors):
                all_vectors[start + j] = vec
            total_tokens += tokens

    return np.asarray(all_vectors, dtype=np.float32), total_tokens


# ── Index building ──


def _load_items(paths: Paths) -> List[Dict[str, Any]]:
    """Load text chunks and image captions into a unified list for indexing."""
    items: List[Dict[str, Any]] = []

    if os.path.exists(paths.chunks_jsonl):
        for r in read_jsonl(paths.chunks_jsonl):
            items.append({
                "id": r["id"],
                "type": "text",
                "doc_id": r.get("doc_id"),
                "doc_name": r.get("doc_name"),
                "page": r.get("page"),
                "content": r.get("text", ""),
            })

    if os.path.exists(paths.captions_jsonl):
        for r in read_jsonl(paths.captions_jsonl):
            items.append({
                "id": r["id"],
                "type": "caption",
                "doc_id": r.get("doc_id"),
                "doc_name": r.get("doc_name"),
                "page": r.get("page"),
                "img_path": r.get("img_path"),
                "content": r.get("caption", ""),
            })

    return items


def build_index(artifacts_dir: str = "artifacts") -> Tuple[int, int]:
    """Build dense (FAISS) and sparse (BM25) indexes.

    Returns (num_items, embedding_dimension).
    """
    paths = Paths(artifacts_dir=artifacts_dir)
    items = _load_items(paths)

    if not items:
        logger.error("Nothing to index — run ingestion and captioning first.")
        return (0, 0)

    texts = [item["content"] for item in items]

    # Dense embeddings via OpenAI (parallel batched)
    logger.info(f"Embedding {len(texts)} items with {EMBED_MODEL}")
    vectors, tokens_used = embed_texts(texts)
    logger.info(f"Embedding done: {tokens_used} tokens used")

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, paths.index_path)

    # Build BM25 index (using rank-bm25 library)
    tokenized = [doc.lower().split() for doc in texts]
    bm25 = BM25Okapi(tokenized)
    with open(paths.bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save metadata (items list + model info)
    meta = {"items": items, "model": EMBED_MODEL}
    with open(paths.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    logger.info(f"Index built: {len(items)} items, {dim}-dim, dense + BM25")
    return (len(items), dim)


# ── Hybrid search ──


def _reciprocal_rank_fusion(
    dense_hits: List[Tuple[int, float]],
    sparse_hits: List[Tuple[int, float]],
    rrf_k: int = 60,
) -> List[Tuple[int, float]]:
    """Combine dense and sparse rankings using Reciprocal Rank Fusion.

    RRF score = 1/(rrf_k + rank) for each ranking list, summed per document.
    Balances semantic and keyword matches without needing score normalization.
    """
    fused: Dict[int, float] = {}

    for rank, (idx, _score) in enumerate(dense_hits):
        fused[idx] = fused.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

    for rank, (idx, _score) in enumerate(sparse_hits):
        fused[idx] = fused.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

    return sorted(fused.items(), key=lambda x: -x[1])


def search(
    query: str,
    k: int = 5,
    artifacts_dir: str = "artifacts",
    metrics: Metrics | None = None,
) -> List[Dict[str, Any]]:
    """Hybrid search: dense (FAISS) + sparse (BM25) with reciprocal rank fusion.

    1. Embed the query with OpenAI
    2. Search FAISS for top-2k semantic matches
    3. Search BM25 for top-2k keyword matches
    4. Fuse both rankings and return the top-k results
    """
    paths = Paths(artifacts_dir=artifacts_dir)

    if not os.path.exists(paths.index_path) or not os.path.exists(paths.meta_path):
        raise FileNotFoundError("Index not found — run build_all first.")

    with open(paths.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    items = meta["items"]

    # Embed query
    query_vec, tokens = embed_texts([query])
    if metrics:
        metrics.add_embedding_cost(EMBED_MODEL, tokens)

    # Dense search (FAISS)
    index = faiss.read_index(paths.index_path)
    scores, indices = index.search(query_vec, min(k * 2, len(items)))
    dense_hits = [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

    # Sparse search (BM25)
    sparse_hits = []
    if os.path.exists(paths.bm25_path):
        with open(paths.bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        bm25_scores = bm25.get_scores(query.lower().split())
        top_indices = bm25_scores.argsort()[-(k * 2):][::-1]
        sparse_hits = [
            (int(i), float(bm25_scores[i])) for i in top_indices if bm25_scores[i] > 0
        ]

    # Fuse rankings
    fused = _reciprocal_rank_fusion(dense_hits, sparse_hits)

    hits: List[Dict[str, Any]] = []
    for idx, rrf_score in fused[:k]:
        hits.append({**items[idx], "score": round(rrf_score, 4)})

    return hits
