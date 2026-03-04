"""Streamlit UI for the Multi-Modal RAG system."""

import os

import streamlit as st
from PIL import Image

from mmrag.rag import answer_question
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──

st.set_page_config(page_title="Multi-Modal RAG")
st.title("Multi-Modal RAG for Datasheets")
st.caption("Ask questions about technical datasheets — answers cite text and images")

# ── Sidebar ──

with st.sidebar:
    st.header("Settings")
    k = st.slider("Results to retrieve", min_value=3, max_value=12, value=5)
    st.markdown("---")
    st.markdown("Requires `OPENAI_API_KEY` environment variable.")

# ── API key check ──

if not os.getenv("OPENAI_API_KEY"):
    st.error("Set `OPENAI_API_KEY` in your environment to use this app.")
    st.stop()

# ── Query ──

question = st.text_input(
    "Ask a question…", placeholder="What is the operating temperature range?"
)

if question and question.strip():
    with st.spinner("Searching and generating answer…"):
        try:
            result = answer_question(question.strip(), k=k)
        except Exception as e:
            st.error(str(e))
            st.stop()

    # Answer
    st.subheader("Answer")
    st.write(result["answer"])

    # Metrics
    m = result["metrics"]
    cols = st.columns(4)
    cols[0].metric("Total latency", f"{m['total_ms']:.0f} ms")
    cols[1].metric("Retrieval", f"{m['retrieval_ms']:.0f} ms")
    cols[2].metric("Generation", f"{m['generation_ms']:.0f} ms")
    cols[3].metric("Cost", f"${m['cost_usd']:.5f}")

    # Sources
    st.subheader("Sources")
    for hit in result["hits"]:
        tag = f"({hit.get('doc_id')}:{hit.get('page')})"
        score = f"score={hit['score']:.4f}"

        if (
            hit["type"] == "caption"
            and hit.get("img_path")
            and os.path.exists(hit["img_path"])
        ):
            cols = st.columns([1, 4])
            with cols[0]:
                st.image(
                    Image.open(hit["img_path"]),
                    caption=tag,
                    width="stretch",
                )
            with cols[1]:
                st.markdown(f"**Image caption** {tag} ({score})")
                st.write(hit["content"])
        else:
            st.markdown(f"**Text** {tag} ({score})")
            content = hit["content"]
            st.write(content[:800] + ("…" if len(content) > 800 else ""))
