# ============================================================
# FILE: reranker.py
# ROLE: Phase 4a — Take the rough candidates returned by
#       vector_db.py and re-score them with a cross-encoder
#       model that reads the query and each candidate together.
#       We're asking for "second opinion" to produce more
#       accurate ranking than cosine similarity alone.
#
# INPUT FILES:
#   - chroma_db/                      ChromaDB from vector_db.py
#   - outputs/embeddings_meta.json    model name used to encode queries
#
# OUTPUT FUNCTIONS:
#   - load_reranker()                 loads the cross-encoder model
#   - retrieve_candidates()           asks ChromaDB for top 50 candidates
#   - rerank()                        returns 50 prompt candidates re-sorted
#                                     by reranker score
# ============================================================

import torch
import chromadb
import json
from sentence_transformers import CrossEncoder, SentenceTransformer

SEED = 42
torch.manual_seed(SEED)


# CONFIGURATION

# recommended model
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# paths to connect to the ChromaDB built by vector_db.py.
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "prompts"
EMBEDDINGS_PATH = "outputs/embeddings_meta.json"

# number candidates to retrieve before reranking
DEFAULT_CANDIDATES = 50



# LOAD MODEL
def load_reranker(model_name=RERANKER_MODEL_NAME):
    """
    Download and load the cross-encoder model.
    """
    print("Loading reranker model:", model_name)
    reranker = CrossEncoder(model_name)

    return reranker



# RETRIEVE CANDIDATES
def retrieve_candidates(collection, encoder, query_text, n=DEFAULT_CANDIDATES):
    """
    Embed the query and retrieve the N closest prompts from ChromaDB
    by cosine similarity. Returns a list of candidate dicts.
    """

    # embed the query into a vector
    query_vector = encoder.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    # find the 50 closest prompts by cosine similarity
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    # unpack the ChromaDB response and build the same dict format
    candidates = []
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    rank = 1
    for i in range(len(ids)):
        candidate = {
            "id": ids[i],
            "content": documents[i],
            "metadata": metadatas[i],
            "similarity": 1.0 - float(distances[i]),
            "rank": rank,
        }
        candidates.append(candidate)
        rank = rank + 1

    return candidates



# RERANK
def rerank(reranker, query_text, candidates):
    """
    Re-score and re-sort a list of candidate prompts for a given query.

    reranker: CrossEncoder — the loaded model (from load_reranker)
    query_text: str — the user's search query
    candidates: list[dict] — 50 prompts retrieved from ChromaDB
    """

    # build input pairs for cross-encoder [query, 1/50 candidate prompts]
    pairs = []
    for candidate in candidates:
        title = candidate["metadata"].get("title", "")
        content = candidate["content"]
        tags = candidate["metadata"].get("tags", "")
        text = title + ". " + content + " " + tags
        pair = [query_text, text]
        pairs.append(pair)

    # score all pairs - receive list of raw relevance scores
    scores = reranker.predict(pairs)

    # attach each score back to its candidate dict
    for candidate, score in zip(candidates, scores):
        candidate["reranker_score"] = float(score)

    # sort candidates by reranker relevance score
    reranked = sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)

    # assign new rank numbers, starting with 1
    new_rank = 1
    for candidate in reranked:
        candidate["reranker_rank"] = new_rank
        new_rank = new_rank + 1

    # output sorted dicts with two new keys: "reranker_score" and "reranker_rank"
    return reranked



# PRINT RESULTS
def print_reranked_results(results, top_n=5):
    """
    Print a table of the top N results after reranking.
    Shows both the original cosine rank and the new reranker rank
    to see how much the ordering changed.
    """

    print(f"\n{'Rank':>4}  {'Was':>4}  {'Score':>7}  {'ID':<10}  {'Category':<18}  Title")
    for r in results[:top_n]:
        original_rank = r.get("rank", "?") # rank by cosine similarity
        reranker_rank = r.get("reranker_rank", "?") # new rank after reranking, returns "?" when not available
        score = round(r.get("reranker_score", 0), 3) # 0 when no reranker relevance score exists
        prompt_id = r.get("id", "")
        category = r.get("metadata", {}).get("category", "")[:18]
        title = r.get("metadata", {}).get("title", "")[:40]
        print(f"{reranker_rank:>4}  {original_rank:>4}  {score:>7}  {prompt_id:<10}  {category:<18}  {title}")



def main():
    print("RERANKER")

    # call the loading reranker function in the variable
    reranker = load_reranker()

    # connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # load the embedding model from embeddings_meta.json
    with open(EMBEDDINGS_PATH) as f:
        embedding_meta = json.load(f)
    encoder = SentenceTransformer(embedding_meta["model_name"])

    sample_queries = [
        "summarise a long document",
        "write a job application cover letter",
    ]

    # for every sample query
    for query_text in sample_queries:
        print("\nQuery:", query_text)

        candidates = retrieve_candidates(collection, encoder, query_text)

        print("\nBEFORE reranking (top 5):")
        print_reranked_results(candidates, top_n=5)

        reranked = rerank(reranker, query_text, candidates)

        print("\nAFTER reranking (top 5):")
        print_reranked_results(reranked, top_n=5)

    print("\nReranker is done. Import load_reranker(), retrieve_candidates() and rerank() from this file")


if __name__ == "__main__":
    main()
