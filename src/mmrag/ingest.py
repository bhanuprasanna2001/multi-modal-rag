"""PDF ingestion: extract text chunks and images from PDFs using parallel processing."""

from __future__ import annotations

import os
import sys
from io import BytesIO
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger
from PIL import Image
from tqdm import tqdm
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

from .utils import ensure_dir, file_stem, stable_hash, chunk_text, Paths, save_jsonl



def _init_worker():
    """Suppress MuPDF C-level stderr warnings in worker processes."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)


# ── Per-page helpers (run inside worker processes) ──


def _extract_images(
    page: fitz.Page, out_dir: str, doc_id: str, page_num: int
) -> List[Dict[str, Any]]:
    """Pull embedded images from a PDF page and save as normalized RGB PNGs."""
    records: List[Dict[str, Any]] = []
    MIN_IMG_SIZE = 50

    for idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            pix = page.parent.extract_image(xref)
            img_bytes = pix["image"]
        except Exception:
            continue

        fname = f"{doc_id}_p{page_num:03d}_{idx:02d}.png"
        img_dir = os.path.join(out_dir, doc_id)
        ensure_dir(img_dir)
        fpath = os.path.join(img_dir, fname)

        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            # Skip tiny images (icons, dots, decorative borders)
            if img.width < MIN_IMG_SIZE or img.height < MIN_IMG_SIZE:
                continue
            img.save(fpath, format="PNG")
        except Exception:
            continue

        records.append({
            "id": stable_hash(f"img|{doc_id}|{page_num}|{idx}"),
            "type": "image",
            "doc_id": doc_id,
            "doc_name": f"{doc_id}.pdf",
            "page": page_num,
            "img_path": fpath,
        })

    return records


def _process_one_pdf(
    pdf_path: str, images_dir: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single PDF — returns (text_chunks, image_records).

    Runs in a separate process for parallelism.
    """
    doc_name = os.path.basename(pdf_path)
    doc_id = file_stem(doc_name)
    chunks: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return chunks, images

    if doc.is_encrypted:
        doc.close()
        return chunks, images

    for page_idx in range(len(doc)):
        try:
            page = doc[page_idx]
        except Exception:
            continue

        page_num = page_idx + 1

        # Text extraction + chunking
        try:
            text = (page.get_text("text") or "").strip()
        except Exception:
            text = ""

        if text:
            for chunk_idx, chunk in enumerate(chunk_text(text)):
                chunks.append({
                    "id": stable_hash(f"txt|{doc_id}|{page_num}|{chunk_idx}"),
                    "type": "text",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": page_num,
                    "text": chunk,
                })

        # Image extraction
        try:
            images.extend(_extract_images(page, images_dir, doc_id, page_num))
        except Exception:
            pass

    doc.close()
    return chunks, images


# ── Main entry point ──


def ingest_pdfs(
    data_dir: str = "data", artifacts_dir: str = "artifacts"
) -> Dict[str, int]:
    """Extract text chunks and images from all PDFs in parallel.

    Uses ProcessPoolExecutor across CPU cores for ~4-8x speedup.
    Returns counts of extracted chunks and images.
    """
    paths = Paths(data_dir=data_dir, artifacts_dir=artifacts_dir)

    pdf_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    )
    logger.info(f"Found {len(pdf_files)} PDFs in {data_dir}")

    # Configurable limit for development (default: process all)
    limit_str = os.getenv("INGEST_PDF_LIMIT", "")
    limit = int(limit_str) if limit_str.isdigit() else len(pdf_files)
    if limit < len(pdf_files):
        logger.info(f"Processing first {limit} PDFs (set INGEST_PDF_LIMIT to change)")
    pdf_files = pdf_files[:limit]

    chunk_records: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []

    workers = min(os.cpu_count() or 4, len(pdf_files))

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(_process_one_pdf, p, paths.images_dir): p
            for p in pdf_files
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Ingesting PDFs"
        ):
            chunks, images = future.result()
            chunk_records.extend(chunks)
            image_records.extend(images)

    save_jsonl(chunk_records, paths.chunks_jsonl)
    save_jsonl(image_records, paths.images_jsonl)

    logger.info(
        f"Ingestion done: {len(chunk_records)} text chunks, "
        f"{len(image_records)} images from {len(pdf_files)} PDFs"
    )
    return {
        "chunks": len(chunk_records),
        "images": len(image_records),
        "pdfs": len(pdf_files),
    }
