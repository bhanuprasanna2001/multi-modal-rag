from __future__ import annotations
import os
from typing import List, Dict, Any
from loguru import logger
from PIL import Image
from transformers import pipeline
from .utils import Paths, read_jsonl, save_jsonl


# Small, CPU-friendly image captioning model
# blip-image-captioning-base is relatively light; replace if needed
DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


def generate_captions(model_name: str = DEFAULT_MODEL, artifacts_dir: str = "artifacts") -> int:
    paths = Paths(artifacts_dir=artifacts_dir)
    images_manifest = os.path.join(paths.artifacts_dir, "images.jsonl")
    if not os.path.exists(images_manifest):
        logger.error("images.jsonl not found. Run ingestion first.")
        return 0

    captioner = pipeline("image-to-text", model=model_name)

    records: List[Dict[str, Any]] = []
    for rec in read_jsonl(images_manifest):
        img_path = rec.get("img_path")
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            # Ensure RGB 3-channel image for the processor
            im = Image.open(img_path).convert("RGB")
            # Some transformers versions expect positional 'inputs'; pass the PIL image positionally
            cap = captioner(im, max_new_tokens=30)
            caption = cap[0]["generated_text"].strip()
        except Exception as e:
            logger.warning(f"Caption failed for {img_path}: {e}")
            continue
        records.append({
            "id": rec["id"],
            "type": "caption",
            "doc_id": rec.get("doc_id"),
            "doc_name": rec.get("doc_name"),
            "page": rec.get("page"),
            "img_path": img_path,
            "caption": caption,
        })

    save_jsonl(records, paths.captions_jsonl)
    logger.info(f"Generated {len(records)} captions")
    return len(records)


if __name__ == "__main__":
    generate_captions()
