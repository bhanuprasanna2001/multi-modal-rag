from __future__ import annotations
import os
from typing import List, Dict, Any
from loguru import logger
from PIL import Image
import fitz  # PyMuPDF
from .utils import ensure_dir, file_stem, stable_hash, Paths, save_jsonl
import re


def _extract_images_from_page(page: fitz.Page, out_dir: str, doc_id: str, page_num: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    image_list = page.get_images(full=True)
    for idx, img in enumerate(image_list):
        xref = img[0]
        try:
            pix = page.parent.extract_image(xref)
            img_bytes = pix["image"]
            ext = pix.get("ext", "png")
        except Exception as e:
            logger.warning(f"Failed extracting image xref={xref} on page {page_num}: {e}")
            continue
        fname = f"{doc_id}_p{page_num:03d}_{idx:02d}.{ext}"
        out_path = os.path.join(out_dir, doc_id)
        ensure_dir(out_path)
        fpath = os.path.join(out_path, fname)
        try:
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            # normalize/sanitize by re-saving as PNG thumbnail-friendly
            try:
                im = Image.open(fpath)
                im.convert("RGB").save(fpath)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Failed writing image {fpath}: {e}")
            continue
        records.append({
            "id": stable_hash(f"img|{doc_id}|{page_num}|{idx}|{fpath}"),
            "type": "image",
            "doc_id": doc_id,
            "page": page_num,
            "img_path": fpath,
        })
    return records


SPEC_PATTERNS = [
    re.compile(r"(-?\d+\s*[°º]?[CF]\s*(?:to|-|–)\s*-?\d+\s*[°º]?[CF])", re.IGNORECASE),  # 0°C to 70°C
    re.compile(r"(operating|storage)\s+temp", re.IGNORECASE),
]


def _extract_spec_lines(text: str) -> List[str]:
    lines = []
    for ln in text.splitlines():
        if any(p.search(ln) for p in SPEC_PATTERNS):
            # Basic cleanliness
            cleaned = re.sub(r"\s+", " ", ln).strip()
            if 3 <= len(cleaned) <= 300:
                lines.append(cleaned)
    return lines


def ingest_pdfs(data_dir: str = "data", artifacts_dir: str = "artifacts") -> Dict[str, int]:
    paths = Paths(data_dir=data_dir, artifacts_dir=artifacts_dir)
    text_records: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []

    pdf_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]
    pdf_files.sort()
    logger.info(f"Found {len(pdf_files)} PDFs in {data_dir}")
    limit_env = os.getenv("INGEST_PDF_LIMIT")
    limit = int(limit_env) if (limit_env and limit_env.isdigit()) else 10
    logger.info(f"Processing first {limit} PDFs (override via INGEST_PDF_LIMIT env var)")

    spec_records: List[Dict[str, Any]] = []

    for pdf in pdf_files[:limit]:
        doc_name = os.path.basename(pdf)
        logger.info(f"Processing PDF: {doc_name}")
        logger.info(f"PDF path: {pdf}")
        doc_id = file_stem(doc_name)
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            logger.error(f"Failed opening {pdf}: {e}")
            continue
        for i in range(len(doc)):
            page = doc[i]
            page_num = i + 1
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                text_records.append({
                    "id": stable_hash(f"txt|{doc_id}|{page_num}"),
                    "type": "text",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": page_num,
                    "text": text,
                })
                # extract candidate spec lines
                for spec_line in _extract_spec_lines(text):
                    spec_records.append({
                        "id": stable_hash(f"spec|{doc_id}|{page_num}|{spec_line}"),
                        "type": "spec",
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "page": page_num,
                        "spec": spec_line,
                    })
            # images
            imgs = _extract_images_from_page(page, paths.images_dir, doc_id, page_num)
            for r in imgs:
                r["doc_name"] = doc_name
            image_records.extend(imgs)
        doc.close()

    # Save artifacts
    save_jsonl(text_records, paths.texts_jsonl)
    # Also store raw image list for downstream captioning convenience
    images_manifest = os.path.join(paths.artifacts_dir, "images.jsonl")
    save_jsonl(image_records, images_manifest)
    if spec_records:
        save_jsonl(spec_records, paths.specs_jsonl)
        logger.info(f"Extracted {len(spec_records)} spec candidate lines")

    logger.info(
        f"Ingestion completed: pages with text={len(text_records)}, images extracted={len(image_records)}"
    )
    return {
        "text_pages": len(text_records),
        "images": len(image_records),
    "pdfs": len(pdf_files),
    "spec_lines": len(spec_records),
    }


if __name__ == "__main__":
    ingest_pdfs()
