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

# Reproducibility — fixing seeds so the same input produces the same vectors every run.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================================================================
# SECTION 1 — CONFIGURATION
# ==================================================================

# Paths (these match the folder layout from preprocessing.py).
CLEANED_INPUT_PATH = "outputs/cleaned_data.json"
EMBEDDINGS_OUTPUT_PATH = "outputs/embeddings.npy"
IDS_OUTPUT_PATH = "outputs/embeddings_ids.json"
META_OUTPUT_PATH = "outputs/embeddings_meta.json"
EMBEDDINGS_BGE_OUTPUT_PATH = "outputs/embeddings_bge.npy"

# Whether to also run the second embedding model (for the A/B comparison).
RUN_SECOND_MODEL = True

# Embedding models. MiniLM is the primary one recommended by the LEAF
# brief and the project guide. BGE is a similar-size alternative used
# only to compare quality in the evaluation phase.
PRIMARY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SECONDARY_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Number of texts encoded together in one forward pass.
BATCH_SIZE = 64

# When True, output vectors have unit length (L2 norm = 1).
# This makes cosine similarity equivalent to a simple dot product
# and matches the cosine-space setup used in ChromaDB (Phase 3).
NORMALIZE_EMBEDDINGS = True

# Use GPU if available (e.g. on Colab), otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" ##??????


# ==================================================================
# SECTION 2 — LOAD CLEANED DATA
# ==================================================================

def load_cleaned_data(path):
    # Loads cleaned_data.json produced by preprocessing.py.
    # Returns a list of records (each record is a dict).

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("Loaded", len(records), "cleaned records from", path)
    print("Fields available:", sorted(records[0].keys()))
    print("Example record id:", records[0]["id"])
    print("Example text_to_embed (first 200 chars):")
    print("   ", records[0]["text_to_embed"][:200])
    return records


# ==================================================================
# SECTION 3 — PREPARE INPUT TEXTS
# ==================================================================

def prepare_inputs(records):
    # Splits records into two parallel lists: texts and ids.
    # texts[i] and ids[i] always refer to the same prompt — this
    # alignment is what lets us map vectors back to prompts later.

    texts = []
    ids = []
    skipped = 0

    for record in records:
        text = record.get("text_to_embed", "")
        record_id = record.get("id", "")
        # Skip empty texts (defensive — preprocessing should already filter these).
        if not text or not text.strip():
            skipped += 1
            continue
        texts.append(text)
        ids.append(record_id)

    if skipped:
        print("Skipped", skipped, "records with empty text_to_embed")
    print("Prepared", len(texts), "texts for encoding")

    # Length statistics — useful to confirm the model's 256-token
    # context window will fit most texts without truncation.
    lengths = [len(t) for t in texts]
    print("Text length (chars): min =", min(lengths),
          ", mean =", sum(lengths) // len(lengths),
          ", max =", max(lengths))
    print("First 3 IDs:", ids[:3])

    return texts, ids


# ==================================================================
# SECTION 4 — ENCODE TEXTS
# ==================================================================

def encode_texts(model_name, texts):
    # Loads a sentence-transformer model and encodes all texts.
    # Returns a numpy array of shape (n_texts, embedding_dim).

    print("Loading model:", model_name)
    print("Device:", DEVICE)
    start = time.time()
    model = SentenceTransformer(model_name, device=DEVICE)
    print("Model loaded in", round(time.time() - start, 1), "s")

    # Print model properties so we know what we're working with.
    embed_dim = model.get_embedding_dimension()
    max_seq_len = model.get_max_seq_length()
    print("Embedding dimension:", embed_dim)
    print("Max sequence length (tokens):", max_seq_len)

    print("Encoding", len(texts), "texts (batch_size =", BATCH_SIZE, ")...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )
    elapsed = time.time() - start

    # Cast to float32 — saves space and matches what ChromaDB expects.
    embeddings = embeddings.astype(np.float32)

    print("Encoding finished in", round(elapsed, 1), "s")
    print("Speed:", int(len(texts) / elapsed), "texts/s")
    print("Output shape:", embeddings.shape)

    # Sanity checks: no NaNs, and (if normalised) vectors should have unit length.
    n_nans = int(np.isnan(embeddings).sum())
    print("NaN values in output:", n_nans)

    sample_norms = np.linalg.norm(embeddings[:5], axis=1)
    print("L2 norms of first 5 vectors:", sample_norms)

    print("First vector preview (first 8 dims):", embeddings[0][:8])

    return embeddings


# ==================================================================
# SECTION 5 — SAVE OUTPUTS
# ==================================================================

def save_outputs(embeddings, ids, model_name):
    # Saves three files:
    #   1) embeddings.npy        — the (n, d) matrix of vectors
    #   2) embeddings_ids.json   — prompt IDs in the same order as the rows
    #   3) embeddings_meta.json  — info about this run (model, seed, etc.)

    os.makedirs("outputs", exist_ok=True)

    # Sanity check — if rows and ids don't match, later lookups would break.
    print("Sanity check: embeddings rows =", embeddings.shape[0],
          ", ids count =", len(ids),
          "(should be equal)")

    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)
    print("Saved", embeddings.shape, "matrix ->", EMBEDDINGS_OUTPUT_PATH)

    with open(IDS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)
    print("Saved", len(ids), "IDs ->", IDS_OUTPUT_PATH)

    meta = {
        "model_name": model_name,
        "n_records": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "normalised": NORMALIZE_EMBEDDINGS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "device": DEVICE,
    }
    with open(META_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved run metadata ->", META_OUTPUT_PATH)
    print("Run metadata:", meta)


# ==================================================================
# SECTION 6 — MAIN ORCHESTRATOR
# ==================================================================

def main():
    pipeline_start = time.time()

    print("=" * 60)
    print("Phase 2: Embeddings")
    print("=" * 60)

    # STEP 1: Load the cleaned dataset built in Phase 1.
    print("\n[1/4] Loading cleaned data")
    records = load_cleaned_data(CLEANED_INPUT_PATH)

    # STEP 2: Pull aligned (texts, ids) lists.
    print("\n[2/4] Preparing input texts")
    texts, ids = prepare_inputs(records)

    # STEP 3: Encode with the primary model and save the outputs
    # used by Phase 3 (vector_db.py).
    print("\n[3/4] Encoding with primary model (" + PRIMARY_MODEL_NAME + ")")
    embeddings = encode_texts(PRIMARY_MODEL_NAME, texts)
    save_outputs(embeddings, ids, PRIMARY_MODEL_NAME)

    # STEP 4: Optional — encode with a second model for the A/B comparison.
    if RUN_SECOND_MODEL:
        print("\n[4/4] Encoding with secondary model (" + SECONDARY_MODEL_NAME + ")")
        embeddings_bge = encode_texts(SECONDARY_MODEL_NAME, texts)
        np.save(EMBEDDINGS_BGE_OUTPUT_PATH, embeddings_bge)
        print("  Saved", embeddings_bge.shape, "matrix ->", EMBEDDINGS_BGE_OUTPUT_PATH)
    else:
        print("\n[4/4] Skipping secondary model (RUN_SECOND_MODEL = False)")

    total = time.time() - pipeline_start
    print("\n✓ Embeddings phase complete in", round(total, 1), "s.")


# ==================================================================
# ENTRY POINT
# ==================================================================
if __name__ == "__main__":
    main()