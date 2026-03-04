"""Image captioning: describe extracted images using OpenAI GPT-4o-mini vision."""

from __future__ import annotations

import os
import base64
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm
from openai import OpenAI

from .utils import Paths, read_jsonl, save_jsonl

CAPTION_PROMPT = (
    "Describe this technical image in one detailed paragraph. "
    "Focus on: component labels, pin names/numbers, values, part numbers, "
    "connector types, and any text visible in the image. "
    "Be specific — include every label and number you can read."
)


def _caption_one(client: OpenAI, img_path: str, model: str) -> str:
    """Send a single image to OpenAI vision and get a caption back."""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "low",  # cheaper, fast, sufficient for datasheets
                }},
            ],
        }],
        max_tokens=300,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def generate_captions(artifacts_dir: str = "artifacts", workers: int = 20) -> int:
    """Caption all extracted images in parallel using OpenAI vision.

    Uses ThreadPoolExecutor with `workers` threads for concurrent API calls.
    Returns the number of successfully captioned images.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for image captioning.")

    paths = Paths(artifacts_dir=artifacts_dir)
    if not os.path.exists(paths.images_jsonl):
        logger.error("images.jsonl not found — run ingestion first.")
        return 0

    client = OpenAI()
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    image_records = read_jsonl(paths.images_jsonl)

    # Filter to images that actually exist on disk
    valid = [r for r in image_records if os.path.exists(r.get("img_path", ""))]
    logger.info(f"Captioning {len(valid)} images with {model} ({workers} workers)")

    captions: List[Dict[str, Any]] = []

    def _process(rec: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            caption = _caption_one(client, rec["img_path"], model)
            return {
                "id": rec["id"],
                "type": "caption",
                "doc_id": rec.get("doc_id"),
                "doc_name": rec.get("doc_name"),
                "page": rec.get("page"),
                "img_path": rec["img_path"],
                "caption": caption,
            }
        except Exception as e:
            logger.warning(f"Caption failed for {rec['img_path']}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, rec): rec for rec in valid}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Captioning images"
        ):
            result = future.result()
            if result:
                captions.append(result)

    save_jsonl(captions, paths.captions_jsonl)
    logger.info(f"Captioned {len(captions)} of {len(valid)} images")
    return len(captions)
