LEAF
---

## Requirements

- Python 3.13
- Install all dependencies in Terminal:

```
pip install -r requirements.txt
```
---

## Execution Order

Run the files in this order. Each step produces output files that the next step uses.

```
Step 1 -> preprocessing.py
Step 2 -> embeddings.py
Step 3 -> vector_db.py
Step 4 -> reranker.py  (uses vector_db internally)
Step 5 -> metadata_fusion.py
Step 6 -> evaluation.py
Step 7 -> streamlit run demo.py
```
| Step | Command | What it does | Output |
|------|---------|--------------|--------|
| 1 | `python` [`preprocessing.py`](preprocessing.py) | Loads dataset, cleans and normalises prompts | [`cleaned_data.json`](outputs/cleaned_data.json), [`metadata_normalised.json`](outputs/metadata_normalised.json) |
| 2 | `python` [`embeddings.py`](embeddings.py) | Encodes all prompts with a sentence-transformer | [`embeddings.npy`](outputs/embeddings.npy), [`embedding_ids.json`](outputs/embedding_ids.json) |
| 3 | `python` [`vector_db.py`](vector_db.py) | Builds ChromaDB index over the embeddings | [`chroma_db/`](chroma_db/) (persistent index) |
| 4 | `python` [`reranker.py`](reranker.py) | Cross-encoder reranking of top candidates | prints ranked results to stdout |
| 5 | `python` [`metadata_fusion.py`](metadata_fusion.py) | Blends semantic score with metadata signals; tunes α, β, γ via Bayesian optimisation | [`best_weights.json`](outputs/best_weights.json) |
| 6 | `python` [`evaluation.py`](evaluation.py) | Runs A/B/C comparison, computes Precision@K and MRR | prints results table |
| 7 | `streamlit run` [`src/demo.py`](demo.py) | Launches the web demo in your browser | opens at `http://localhost:8501` |
> **Optional:** `python` [`embeddings_vis.py`](embeddings_vis.py) — generates t-SNE/UMAP visualisations of the embedding space. Run after Step 2.

---

## Dataset

Raw dataset file at:
[LEAF-promptkaban-dataset/dataset.json](LEAF-promptkaban-dataset/dataset.json)


`preprocessing.py` will look for it in the same directory.
