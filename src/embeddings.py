# ============================================================
# FILE: embeddings.py
# ROLE: Phase 2 — Convert every cleaned prompt into a dense
#       semantic vector using a pre-trained sentence-transformer
#       model. Outputs the embedding matrix and aligned IDs used
#       by vector_db.py in Phase 3.
#
# INPUT FILES:
#   - outputs/cleaned_data.json         from preprocessing.py
#
# OUTPUT FILES:
#   - outputs/embeddings.npy            primary model vectors (MiniLM, 384-dim)
#   - outputs/embeddings_ids.json       prompt IDs aligned with embedding rows
#   - outputs/embeddings_meta.json      run metadata (model, dim, seed, ...)
#   - outputs/embeddings_bge.npy        secondary model vectors for A/B
# RUN:  python embeddings.py
# ============================================================

import json
import os
import random
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CONFIGURATION
CLEANED_INPUT_PATH = "outputs/cleaned_data.json"
EMBEDDINGS_OUTPUT_PATH = "outputs/embeddings.npy"
IDS_OUTPUT_PATH = "outputs/embeddings_ids.json"
META_OUTPUT_PATH = "outputs/embeddings_meta.json"
EMBEDDINGS_BGE_OUTPUT_PATH = "outputs/embeddings_bge.npy"

PRIMARY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SECONDARY_MODEL_NAME = "BAAI/bge-small-en-v1.5"



# LOAD CLEANED DATA
def load_cleaned_data(path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    print("Loaded", len(records), "cleaned records")
    return records


# PREPARE INPUT TEXTS
def prepare_inputs(records):
    texts = []
    ids = []
    skipped = 0

    for record in records:
        text = record.get("text_to_embed", "").strip()
        if not text:
            skipped += 1
            continue
        texts.append(text)
        ids.append(record["id"])

    if skipped:
        print("Skipped", skipped, "records with empty text_to_embed")
    print("Prepared", len(texts), "texts")

    # Char lengths, rough proxy for the 256-token context window.
    lengths = [len(t) for t in texts]
    print("Text length (chars): min", min(lengths),
          "mean", sum(lengths) // len(lengths),
          "max", max(lengths))

    return texts, ids

# ENCODE TEXTS
def encode_texts(model_name, texts):
    print("Loading model:", model_name)
    model = SentenceTransformer(model_name)

    print("Encoding", len(texts), "texts...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start

    embeddings = embeddings.astype(np.float32)

    print("Done in", round(elapsed, 1), "s. Shape:", embeddings.shape)
    return embeddings


# SAVE OUTPUTS
def save_outputs(embeddings, ids, model_name):
    os.makedirs("outputs", exist_ok=True)

    assert embeddings.shape[0] == len(ids)

    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)
    print("Saved", embeddings.shape, "->", EMBEDDINGS_OUTPUT_PATH)

    with open(IDS_OUTPUT_PATH, "w") as f:
        json.dump(ids, f)
    print("Saved", len(ids), "IDs ->", IDS_OUTPUT_PATH)

    meta = {
        "model_name": model_name,
        "n_records": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "normalised": True,
        "batch_size": 64,
        "seed": SEED,
    }
    with open(META_OUTPUT_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta ->", META_OUTPUT_PATH)


# MAIN
def main():
    print("EMBEDDINGS:")

    records = load_cleaned_data(CLEANED_INPUT_PATH)
    texts, ids = prepare_inputs(records)

    print("\nPrimary model:", PRIMARY_MODEL_NAME)
    embeddings = encode_texts(PRIMARY_MODEL_NAME, texts)
    save_outputs(embeddings, ids, PRIMARY_MODEL_NAME)

    print("\nSecondary model:", SECONDARY_MODEL_NAME)
    embeddings_bge = encode_texts(SECONDARY_MODEL_NAME, texts)
    np.save(EMBEDDINGS_BGE_OUTPUT_PATH, embeddings_bge)
    print("Saved", embeddings_bge.shape, "->", EMBEDDINGS_BGE_OUTPUT_PATH)


if __name__ == "__main__":
    main()
