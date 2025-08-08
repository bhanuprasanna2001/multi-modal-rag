# Architecture

- Ingestion: PyMuPDF to extract page text and images.
- Captioning: Transformers pipeline (BLIP base) to caption images.
- Indexing: SentenceTransformer (all-MiniLM-L6-v2) to embed text+captions; FAISS IP index.
- Retrieval: cosine via normalized inner product; unified mixed-type retrieval.
- Generation: Strictly from context; OpenAI if available else conservative local fallback.
- UI: Streamlit showing answer, citations, and sources with image thumbs.
