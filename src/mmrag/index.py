from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Tuple
from loguru import logger
import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import faiss as faiss_mod
    _FAISS_OK = True
else:
    try:
        import faiss as faiss_mod  # type: ignore
        _FAISS_OK = True
    except Exception:
        faiss_mod = None  # type: ignore
        _FAISS_OK = False
from sentence_transformers import SentenceTransformer
from .utils import Paths, read_jsonl, save_jsonl


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_items(paths: Paths) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if os.path.exists(paths.texts_jsonl):
        for r in read_jsonl(paths.texts_jsonl):
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
    # Optional spec lines
    if os.path.exists(paths.specs_jsonl):
        for r in read_jsonl(paths.specs_jsonl):
            items.append({
                "id": r["id"],
                "type": "spec",
                "doc_id": r.get("doc_id"),
                "doc_name": r.get("doc_name"),
                "page": r.get("page"),
                "content": r.get("spec", ""),
            })

    return items


def build_index(artifacts_dir: str = "artifacts", model_name: str = EMBED_MODEL) -> Tuple[int, int]:
    paths = Paths(artifacts_dir=artifacts_dir)
    items = _load_items(paths)
    if not items:
        logger.error("No items to index. Run ingestion and captioning first.")
        return (0, 0)

    model = SentenceTransformer(model_name)
    texts = [it["content"] for it in items]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype=np.float32)

    if _FAISS_OK:
        assert faiss_mod is not None
        index = faiss_mod.IndexFlatIP(vecs.shape[1])  # type: ignore[attr-defined]
        index.add(vecs)  # type: ignore[call-arg]
        faiss_mod.write_index(index, paths.index_path)  # type: ignore[attr-defined]
    else:
        # Save vectors as numpy for fallback search
        np.save(paths.index_path + ".npy", vecs)
    meta = {
        "items": items,
        "model": model_name,
        "normalize": True,
    "metric": "ip",
    "faiss": _FAISS_OK,
    }
    with open(paths.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    logger.info(f"Built index with {len(items)} items, dim={vecs.shape[1]}")
    return (len(items), vecs.shape[1])


def search(query: str, k: int = 6, artifacts_dir: str = "artifacts", model_name: str = EMBED_MODEL) -> List[Dict[str, Any]]:
    paths = Paths(artifacts_dir=artifacts_dir)
    if not (os.path.exists(paths.index_path) and os.path.exists(paths.meta_path)):
        raise FileNotFoundError("Index not found. Build it first.")

    with open(paths.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    items = meta["items"]
    model = SentenceTransformer(model_name)

    qv = model.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype=np.float32)

    if meta.get("faiss") and _FAISS_OK:
        assert faiss_mod is not None
        index = faiss_mod.read_index(paths.index_path)  # type: ignore[attr-defined]
        D, I = index.search(qv, k)
        scores, idxs = D[0], I[0]
    else:
        vecs = np.load(paths.index_path + ".npy")
        # cosine via normalized inner product; vecs and qv already normalized
        sims = vecs @ qv[0]
        idxs = np.argsort(-sims)[:k]
        scores = sims[idxs]
    hits: List[Dict[str, Any]] = []
    for score, idx in zip(scores, idxs):
        if idx == -1:
            continue
        it = items[idx]
        hits.append({**it, "score": float(score)})
    return hits
