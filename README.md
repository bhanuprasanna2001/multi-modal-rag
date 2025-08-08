# Basic Multi-Modal RAG for Datasheets (Single-Step QA)

Answer simple, factual questions about a set of product/electronics PDFs by retrieving from text and image-derived captions and generating a concise answer with page-level citations.

## Quickstart

1) Create a virtual environment and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Build artifacts from PDFs (text, images, captions, index)

```bash
python -m scripts.build_all
```

3) Run the demo UI

```bash
streamlit run streamlit_app.py
```

4) Ask a question and see the answer with citations and sources.

Optional: set `OPENAI_API_KEY` to use a small hosted model (default `gpt-4o-mini`). Without it, a simple local fallback selects the most relevant line from retrieved context.

## Project Structure

- `data/` — put PDFs here (already populated)
- `artifacts/` — generated: text chunks, images, captions, vector index
- `src/mmrag/` — ingestion, captioning, indexing, RAG code
- `scripts/` — pipeline scripts (`build_all.py`, `qa_cli.py`)
- `eval/` — questions and evaluation script
- `streamlit_app.py` — tiny UI

## Pipeline

1. Ingestion (`mmrag.ingest`):
	- Extract per-page text (page-level chunks) to `artifacts/chunks.jsonl`
	- Extract images to `artifacts/images/<doc_id>/...`
2. Captioning (`mmrag.caption`):
	- Auto-caption all images → `artifacts/captions.jsonl`
3. Indexing (`mmrag.index`):
	- Embed text chunks + captions (all-MiniLM-L6-v2) → FAISS index + metadata
4. Single-Step RAG (`mmrag.rag`):
	- Retrieve top-k mixed results
	- Build compact context with `(DOC:PAGE)` tags
	- Generate answer strictly from context (OpenAI if key set; else local fallback)
5. UI: Shows answer, citations, sources list with thumbnails for image hits

## Evaluation

Run:
```bash
python -m eval.run_eval
```
Outputs JSON with accuracy, average and p95 latency, and image-question success.

Ablation (text-only): temporarily move or rename `artifacts/captions.jsonl` and rebuild the index, then re-run eval.

## Notes & Limits

- Single-step only; page-level chunks; captions can be noisy.
- Performance: uses CPU-friendly models. For faster inference, set `OPENAI_API_KEY`.
- Reproducibility: versions pinned; seeds fixed where applicable.