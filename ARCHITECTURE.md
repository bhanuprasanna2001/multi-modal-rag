# Architecture

## Pipeline Overview

```
PDFs → Ingestion (PyMuPDF, ProcessPoolExecutor)
    ├── Text → 1200-char chunks (200 overlap)
    └── Images → RGB PNGs (in-memory conversion)
                    ↓
        GPT-4o-mini Vision → detailed captions
        (parallel, ThreadPoolExecutor)
                    ↓
Text Chunks + Captions → OpenAI text-embedding-3-small (1536-dim)
                         (parallel batched, batch_size=512)
                    ↓
            ┌───────┴───────┐
        FAISS Index      BM25 Index
        (dense, IP)    (rank-bm25, pkl)
            └───────┬───────┘
                    ↓
     Query → Embed → FAISS search + BM25 search
                    ↓
        Reciprocal Rank Fusion (k=60)
                    ↓
        Top-K context → GPT-4o-mini
                    ↓
        Answer + Citations + Metrics
```

## Modules

| Module | Responsibility |
|---|---|
| `ingest.py` | Parallel PDF parsing via PyMuPDF (ProcessPoolExecutor), sentence-aware chunking, image extraction |
| `caption.py` | Image → technical caption via GPT-4o-mini vision (parallel ThreadPoolExecutor) |
| `index.py` | OpenAI embeddings (parallel batched), FAISS index, rank-bm25, reciprocal rank fusion |
| `rag.py` | Hybrid retrieval → context assembly → GPT-4o-mini generation → cost/latency metrics |
| `utils.py` | Paths, Metrics, Timer, text chunking, JSONL I/O |
| `cli.py` | CLI entry points: `mmrag-build`, `mmrag-query` |

## Models & Cost

| Component | Model | Details | Cost |
|---|---|---|---|
| Embeddings | `text-embedding-3-small` | 1536-dim, parallel batched | $0.02 / 1M tokens |
| Captioning | `gpt-4o-mini` (vision) | detail="low", base64 PNG | $0.15 / 1M input tokens |
| Generation | `gpt-4o-mini` | temperature=0 | $0.15 / 1M in, $0.60 / 1M out |
| Dense index | FAISS `IndexFlatIP` | File-based, no server | Free |
| Sparse index | `rank-bm25` (BM25Okapi) | Pickle-serialized | Free |

## Search Strategy: Hybrid BM25 + Dense

1. **Dense retrieval:** Query → OpenAI embedding → FAISS inner product search → ranked results
2. **Sparse retrieval:** Query → BM25Okapi scoring (rank-bm25 library) → ranked results
3. **Fusion:** Reciprocal rank fusion with k=60 merges both rankings without score normalization

Why hybrid? Dense embeddings excel at semantic similarity but miss exact part numbers, pin names, and spec values. BM25 catches exact matches. RRF combines rankings without needing to normalize incompatible score distributions.

## Metrics Tracking

Every call to `answer_question()` returns a `metrics` dict:
- `retrieval_ms` — time for embedding + FAISS + BM25 + fusion
- `generation_ms` — time for OpenAI completion
- `total_ms` — end-to-end
- `embedding_tokens` — tokens used for query embedding
- `generation_tokens` — prompt + completion tokens
- `cost_usd` — total cost of all API calls in this request

## Key Design Decisions

- **Hybrid search (BM25 + dense + RRF)** — dense misses exact specs; BM25 catches them; RRF combines without score normalization
- **OpenAI for all ML** — eliminates 2.5 GB of local deps (torch, transformers); faster on M1 Pro than local inference
- **Parallel PDF ingestion** — ProcessPoolExecutor across CPU cores; 4-8x faster than serial for 1,000+ PDFs
- **rank-bm25 library** — battle-tested BM25Okapi instead of custom implementation
- **FAISS over vector DBs** — file-based, no server, no migrations; sufficient for corpus size
- **Per-request cost tracking** — built into the response, not bolted on as external observability
