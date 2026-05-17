# ============================================================
# FILE: demo.py
# ROLE: Phase 5b — Streamlit demo for PromptKaban semantic search.
#       Query → retrieve → rerank → fuse → display results.
# RUN:  streamlit run demo.py
# ============================================================

import json, os, sys, chromadb, streamlit as st
from sentence_transformers import SentenceTransformer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

from reranker import load_reranker, rerank, retrieve_candidates
from metadata_fusion import load_metadata, fuse

@st.cache_resource(show_spinner=False)
def load_pipeline():
    enc = SentenceTransformer(json.load(open("outputs/embeddings_meta.json"))["model_name"])
    col = chromadb.PersistentClient(path="chroma_db").get_collection("prompts")
    w   = json.load(open("outputs/best_weights.json"))
    return enc, col, load_reranker(), load_metadata(), w

with st.spinner("Loading models…"):
    encoder, collection, reranker_model, metadata_lookup, weights = load_pipeline()

st.title("🔍 Intelligent Prompt Retrieval")
st.caption("DEMO — LEAF srl · LUISS University · AI Techniques 2026")

query = st.text_input("Enter a search query, then press Search", placeholder="e.g. write a cold email to a potential investor")
k     = st.slider("Results", 1, 20, 5)

if st.button("Search →", type="primary") and query.strip():
    with st.spinner("Searching…"):
        results = fuse(rerank(reranker_model, query, retrieve_candidates(collection, encoder, query)),
                       metadata_lookup, **weights)
    for i, r in enumerate(results[:k]):
        m = r.get("metadata", {})
        with st.container(border=True):
            st.markdown(f"**#{i+1} — {m.get('title', '')}**")
            st.caption(f"{m.get('category','')} › {m.get('subcategory','')}  ·  {m.get('difficulty','')}")
            st.code(r.get("content", ""), language=None, wrap_lines=True)
            st.caption(f"Score: {round(r.get('final_score', 0), 3)}  ·  ♥ {m.get('likes', 0)}  ·  ↺ {m.get('uses', 0)} uses")
