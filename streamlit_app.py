import os, sys
import streamlit as st
from PIL import Image

# Ensure src is importable
REPO_ROOT = os.path.dirname(__file__)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmrag.rag import answer_question

st.set_page_config(page_title="Multi-Modal RAG: Datasheets", layout="wide")

st.title("Basic Multi-Modal RAG for Datasheets")
st.caption("Single-step QA over text pages and image captions with citations")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K", 3, 12, 6)
    artifacts_dir = st.text_input("Artifacts dir", value="artifacts")
    st.markdown("---")
    st.write("Set OPENAI_API_KEY in env to use a hosted LLM; otherwise local heuristic fallback is used.")

q = st.text_input("Ask a question…", placeholder="What is the operating temperature range?")

if st.button("Ask") or (q and st.session_state.get("last_q") != q):
    st.session_state["last_q"] = q
    if not q.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Retrieving and answering…"):
            out = answer_question(q.strip(), k=k, artifacts_dir=artifacts_dir)
        st.subheader("Answer")
        st.write(out["answer"])  # includes citations

        st.subheader("Sources")
        hits = out["hits"]
        for h in hits:
            tag = f"({h.get('doc_id')}:{h.get('page')})"
            if h["type"] == "caption":
                cols = st.columns([1, 4])
                with cols[0]:
                    if h.get("img_path") and os.path.exists(h["img_path"]):
                        im = Image.open(h["img_path"])  # type: ignore
                        st.image(im, caption=tag, use_container_width=True)
                with cols[1]:
                    st.markdown(f"**Image caption {tag}:** {h['content']}")
            else:
                st.markdown(f"**Text {tag}:** {h['content'][:500]}{' …' if len(h['content'])>500 else ''}")
