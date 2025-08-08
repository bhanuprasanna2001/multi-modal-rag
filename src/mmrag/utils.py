from __future__ import annotations
import os
import json
import hashlib
from dataclasses import dataclass
from typing import Iterable, Dict, Any
from loguru import logger

SEED = 42


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def save_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {out_path}")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


@dataclass
class Paths:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    @property
    def texts_jsonl(self) -> str:
        return os.path.join(self.artifacts_dir, "chunks.jsonl")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.artifacts_dir, "images")

    @property
    def captions_jsonl(self) -> str:
        return os.path.join(self.artifacts_dir, "captions.jsonl")

    @property
    def items_jsonl(self) -> str:
        return os.path.join(self.artifacts_dir, "items.jsonl")

    @property
    def index_path(self) -> str:
        return os.path.join(self.artifacts_dir, "index.faiss")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.artifacts_dir, "index_meta.json")

    @property
    def specs_jsonl(self) -> str:
        return os.path.join(self.artifacts_dir, "specs.jsonl")
