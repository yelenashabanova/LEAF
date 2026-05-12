# ============================================================
# FILE: demo.py
# ROLE: Phase 5b — Interactive Streamlit web demo for the
#       LEAF PromptKaban semantic search engine.
#
#       Runs the full pipeline on a user query:
#         1. Retrieve top-50 candidates from ChromaDB (MiniLM)
#         2. Rerank with cross-encoder (ms-marco-MiniLM-L-6-v2)
#         3. Apply metadata fusion with Optuna-tuned weights
#       Displays the top K results as cards with title, category,
#       content preview, and popularity signals.
#
# INPUT FILES:
#   - chroma_db/                       ChromaDB from vector_db.py
#   - outputs/embeddings_meta.json     model name used to encode queries
#   - outputs/metadata_normalised.json normalised popularity signals
#   - outputs/best_weights.json        Optuna-tuned α/β/γ/δ weights
#
# RUN:  streamlit run demo.py
# ============================================================

import html as html_lib
import json
import math
import os
import sys
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

from reranker import load_reranker, rerank
from metadata_fusion import load_metadata, retrieve_candidates, fuse

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

CHROMA_DB_PATH  = os.path.join(SRC_DIR, "chroma_db")
COLLECTION_NAME = "prompts"
EMBEDDINGS_PATH = os.path.join(SRC_DIR, "outputs", "embeddings_meta.json")
WEIGHTS_PATH    = os.path.join(SRC_DIR, "outputs", "best_weights.json")

# ---------------------------------------------------------------
# PAGE CONFIG & STYLE
# ---------------------------------------------------------------

st.set_page_config(page_title="PromptKaban Search", page_icon="🔍", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 860px; }

    /* ── Header ── */
    .pk-wordmark { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
                   text-transform: uppercase; color: #0369a1; margin-bottom: 0.5rem; }
    .pk-title    { font-size: 2.4rem; font-weight: 800; color: #0f172a;
                   letter-spacing: -1px; line-height: 1.1; margin-bottom: 0.4rem; }
    .pk-title span { color: #0369a1; }
    .pk-desc     { font-size: 0.92rem; color: #64748b; max-width: 520px;
                   line-height: 1.6; margin: 0 auto 1rem; }
    .pk-pills    { display: flex; justify-content: center; gap: 0.5rem;
                   flex-wrap: wrap; margin-bottom: 0.25rem; }
    .pk-pill     { display: inline-flex; align-items: center; gap: 5px;
                   background: #f1f5f9; color: #475569; border-radius: 20px;
                   padding: 3px 12px; font-size: 0.75rem; font-weight: 600; }
    .pk-pill.blue { background: #e0f2fe; color: #0369a1; }

    /* ── Divider ── */
    .pk-divider  { border: none; border-top: 1px solid #e2e8f0;
                   margin: 1.6rem 0 1.4rem; }

    /* ── Result cards ── */
    .result-rank  { font-size: 0.7rem; color: #94a3b8; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 2px; }
    .result-title { font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
    .result-cat   { font-size: 0.72rem; color: #94a3b8; margin-bottom: 0.5rem; }
    .result-foot  { font-size: 0.72rem; color: #94a3b8; margin-top: 0.5rem; }
    .score-pill   { background: #0f172a; color: #fff; border-radius: 20px;
                    padding: 4px 13px; font-size: 0.76rem; font-weight: 700;
                    white-space: nowrap; }
    .diff-pill    { border-radius: 20px; padding: 1px 9px; font-size: 0.7rem;
                    font-weight: 600; margin-left: 6px; vertical-align: middle; }
    .diff-beginner     { background:#dcfce7; color:#15803d; }
    .diff-intermediate { background:#fef9c3; color:#a16207; }
    .diff-advanced     { background:#ffedd5; color:#c2410c; }
    .diff-expert       { background:#fce7f3; color:#be185d; }

    /* ── Search input ── */
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
    div[data-testid="stTextInput"] > div > div > input {
        border-radius: 10px !important; font-size: 0.95rem !important;
        padding: 0.7rem 1.1rem !important; border: 1.5px solid #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #0369a1 !important;
        box-shadow: 0 0 0 3px rgba(3,105,161,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# LOAD MODELS at startup — spinner shown every time until cached
# ---------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline():
    with open(EMBEDDINGS_PATH) as f:
        embedding_meta = json.load(f)
    enc = SentenceTransformer(embedding_meta["model_name"])
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col    = client.get_collection(name=COLLECTION_NAME)
    rr     = load_reranker()
    meta   = load_metadata()
    with open(WEIGHTS_PATH) as f:
        w = json.load(f)
    return enc, col, rr, meta, w

with st.spinner("Loading models… (first run only)"):
    encoder, collection, reranker_model, metadata_lookup, weights = load_pipeline()

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------

st.markdown("""
<div style="text-align:center;">
    <div class="pk-wordmark">LEAF srl &nbsp;·&nbsp; LUISS University &nbsp;·&nbsp; AI Techniques 2026</div>
    <div class="pk-title">Search through <span>18,709</span><br>AI prompts</div>
    <div class="pk-desc">
        Semantic search powered by MiniLM embeddings, cross-encoder reranking,
        and Bayesian-optimised metadata fusion.
    </div>
    <div class="pk-pills">
        <span class="pk-pill blue">⚡ Vector + Reranker + Fusion</span>
        <span class="pk-pill">18,709 prompts</span>
        <span class="pk-pill">50 candidates → reranked</span>
    </div>
</div>
<hr class="pk-divider">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# SEARCH FORM — Enter key submits
# ---------------------------------------------------------------

with st.form("search_form", border=False):
    query = st.text_input("", placeholder="e.g.  write a cold email to a potential investor",
                          label_visibility="collapsed")
    col_l, col_r = st.columns([4, 1])
    with col_l:
        k = st.slider("Results", min_value=1, max_value=20, value=5)
    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Search →", type="primary", use_container_width=True)

# ---------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------

if submitted and query.strip():
    with st.spinner("Searching…"):
        candidates = retrieve_candidates(collection, encoder, query.strip())
        reranked   = rerank(reranker_model, query.strip(), candidates)
        results    = fuse(reranked, metadata_lookup,
                          alpha=weights["alpha"], beta=weights["beta"],
                          gamma=weights["gamma"], delta=weights["delta"])
    st.session_state["results"] = results
    st.session_state["last_query"] = query.strip()
    st.session_state["last_k"] = k
    for j in range(k):
        st.session_state[f"exp_{j}"] = False

if "results" in st.session_state:
    results     = st.session_state["results"]
    last_query  = st.session_state["last_query"]
    last_k      = st.session_state["last_k"]

    st.markdown(f"<br>**{last_k} results for:** *{last_query}*", unsafe_allow_html=True)
    st.markdown("")

    for i, r in enumerate(results[:last_k]):
        meta     = r.get("metadata", {})
        title    = meta.get("title",       r.get("id", ""))
        category = meta.get("category",    "")
        subcat   = meta.get("subcategory", "")
        diff     = meta.get("difficulty",  "")
        likes    = int(meta.get("likes", 0))
        uses     = int(meta.get("uses",  0))
        score    = round(r.get("final_score", r.get("reranker_score", 0)), 3)
        full_content = r.get("content", "")
        cat_label    = f"{category} › {subcat}" if subcat else category
        diff_class   = f"diff-{diff.lower()}" if diff else ""
        diff_html    = f'<span class="diff-pill {diff_class}">{diff}</span>' if diff else ""
        safe_content = html_lib.escape(full_content)

        with st.container(border=True):
            head_col, score_col = st.columns([6, 1])
            with head_col:
                st.markdown(
                    f'<div class="result-rank">#{i+1}</div>'
                    f'<div class="result-title">{title}{diff_html}</div>'
                    f'<div class="result-cat">{cat_label}</div>',
                    unsafe_allow_html=True,
                )
            with score_col:
                st.markdown(
                    f'<div style="padding-top:0.3rem;text-align:right;">'
                    f'<span class="score-pill">{score}</span></div>',
                    unsafe_allow_html=True,
                )

            st.code(full_content, language=None, wrap_lines=True)

            st.components.v1.html(f"""
<style>
  .cpbtn {{
    display:inline-flex; align-items:center; gap:6px;
    background:#0f172a; color:#fff; border:none; border-radius:8px;
    padding:7px 16px; font-size:13px; font-weight:600; cursor:pointer;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    transition:background .15s;
  }}
  .cpbtn:hover {{ background:#1e293b; }}
  .cpbtn svg {{ width:14px; height:14px; flex-shrink:0; }}
</style>
<button class="cpbtn" onclick="
  navigator.clipboard.writeText(this.dataset.t).then(()=>{{
    this.innerHTML='<svg viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\'><polyline points=\\'20 6 9 17 4 12\\'/></svg> Copied!';
    setTimeout(()=>{{this.innerHTML=orig}},1800);
  }});" data-t="{safe_content}"
  id="cb{i}">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>
  Copy
</button>
<script>var orig=document.getElementById('cb{i}').innerHTML;</script>
""", height=48)

            st.markdown(
                f'<div class="result-foot">👍 {likes} &nbsp;·&nbsp; 🔁 {uses} uses</div>',
                unsafe_allow_html=True,
            )

elif submitted:
    st.warning("Please enter a search query first.")
