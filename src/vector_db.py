# ============================================================
# FILE: vector_db.py
# ROLE: Phase 3 — Build a persistent ChromaDB index over the
#       produced prompt embeddings and expose a
#       cosine-similarity query function. This is the retrieval
#       backbone of the LEAF semantic search pipeline.
#
# INPUT FILES:
#   - outputs/cleaned_data.json       from preprocessing.py
#   - outputs/embeddings.npy          from embeddings.py
#   - outputs/embeddings_ids.json     from embeddings.py
#   - outputs/embeddings_meta.json    from embeddings.py
#
# OUTPUT FILES:
#   - chroma_db/                      persistent ChromaDB store
#                                     (one collection: COLLECTION_NAME)
#
# ============================================================

import json
import time
import chromadb
import numpy as np
import torch
import shutil
import os
from sentence_transformers import SentenceTransformer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# CONFIG
CLEANED_INPUT_PATH = "outputs/cleaned_data.json"
EMBEDDINGS_INPUT_PATH = "outputs/embeddings.npy"
IDS_INPUT_PATH = "outputs/embeddings_ids.json"
META_INPUT_PATH = "outputs/embeddings_meta.json"

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "prompts"

INSERT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 50
METADATA_FIELDS = [
    "title", "category", "subcategory", "difficulty",
    "likes", "upvotes", "downvotes", "views", "uses",
    "author_reputation", "fork_count", "version",
    "has_placeholders", "language", "target_model", "created_at",
]


# LOAD INPUTS
def load_inputs():
    embeddings = np.load(EMBEDDINGS_INPUT_PATH).astype(np.float32)
    print("Loaded embeddings:", embeddings.shape)

    with open(IDS_INPUT_PATH, encoding="utf-8") as f:
        ids = json.load(f)
    print("Loaded", len(ids), "IDs")

    with open(META_INPUT_PATH, encoding="utf-8") as f:
        run_meta = json.load(f)

    with open(CLEANED_INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)
    print("Loaded", len(records), "cleaned records")

    assert embeddings.shape[0] == len(ids)
    return embeddings, ids, run_meta, records


# CREATE COLLECTION
def create_collection(client):
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# INSERT VECTORS
def build_metadata(record):
    # ChromaDB metadata values must be primitive types

    meta = {}
    for field in METADATA_FIELDS:
        if field in record and record[field] is not None:
            meta[field] = record[field]

    # ChromaDB cannot store lists directly
    if record.get("tags"):
        meta["tags"] = " ".join(record["tags"])

    return meta


def insert_in_batches(collection, embeddings, ids, id_to_record):
    n = len(ids)
    print("Inserting", n, "vectors")

    start = time.time()
    for batch_start in range(0, n, INSERT_BATCH_SIZE):
        batch_end = min(batch_start + INSERT_BATCH_SIZE, n)
        batch_ids = ids[batch_start:batch_end]
        batch_vectors = embeddings[batch_start:batch_end].tolist()

        batch_documents = [id_to_record[pid].get("content", "") for pid in batch_ids]
        batch_metadatas = [build_metadata(id_to_record[pid]) for pid in batch_ids]

        collection.add(
            ids=batch_ids,
            embeddings=batch_vectors,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        print("inserted", batch_end, "/", n)

    print("Done in", round(time.time() - start, 1), "s. Collection size:", collection.count())


# QUERY
def load_query_encoder(run_meta):
    model_name = run_meta["model_name"]
    print("Loading query encoder:", model_name)
    return SentenceTransformer(model_name)


def encode_query(model, query):
    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    return vector


def query(collection, model, query_text, top_k=DEFAULT_TOP_K):
    query_vector = encode_query(model, query_text)

    raw = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    result_ids = raw["ids"][0]
    result_docs = raw["documents"][0]
    result_metas = raw["metadatas"][0]
    result_distances = raw["distances"][0]

    results = []
    for i, pid in enumerate(result_ids):
        results.append({
            "id": pid,
            "content": result_docs[i],
            "metadata": result_metas[i],
            "similarity": 1.0 - float(result_distances[i]),
            "distance": float(result_distances[i]),
            "rank": i + 1,
        })
    return results


# SAMPLE QUERIES
def print_sample_queries(collection, model):
    sample_queries = [
        "write a cold outreach email to a potential investor",
        "balance turn-based combat stats for an RPG",
    ]
    for q in sample_queries:
        print("\nQuery:", q)
        for r in query(collection, model, q, top_k=5):
            print(" ", r["rank"], "|", round(r["similarity"], 3), "|",
                  r["id"], "|", r["metadata"].get("category", ""), "|",
                  r["metadata"].get("title", ""))


# INDEX SUMMARY
def print_index_summary(collection, run_meta):
    sample = collection.peek(limit=1)
    keys = sorted(sample["metadatas"][0].keys())

    print("\nINDEX SUMMARY")
    print("Collection      :", COLLECTION_NAME)
    print("Persisted at    :", CHROMA_DB_PATH)
    print("Distance space  : cosine (HNSW)")
    print("Model           :", run_meta["model_name"])
    print("Dimension       :", run_meta["embedding_dim"])
    print("Vectors         :", collection.count())
    print("Metadata fields :", len(keys))
    print("Keys            :", keys)


# MAIN
def main():
    print("VECTOR DB")

    print("\nLoading inputs")
    embeddings, ids, run_meta, records = load_inputs()

    # embeddings.npy and cleaned_data.json may not be in the same row order,
    # so we look up records by ID when assembling each batch.
    id_to_record = {}
    for record in records:
        record_id = record["id"]
        id_to_record[record_id] = record

    # vector.db clean up
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    print("\nInitialising ChromaDB at", CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = create_collection(client)

    print("\nIndexing vectors")
    insert_in_batches(collection, embeddings, ids, id_to_record)

    print("\nRunning sample queries")
    model = load_query_encoder(run_meta)
    print_sample_queries(collection, model)

    print_index_summary(collection, run_meta)


if __name__ == "__main__":
    main()