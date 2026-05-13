# ============================================================
# FILE: evaluation.py
# ROLE: Phase 5 — Measure and compare the search quality of
#       three pipeline configurations using the manually
#       annotated evaluation set (eval_queries.json).
#
#       The three configurations compared are:
#         A) Vector search only (cosine similarity, no reranker)
#         B) Vector search + reranker (cross-encoder)
#         C) Full pipeline (reranker + metadata fusion)
#
#       Two metrics are computed for each configuration:
#         Precision@5 — how many of the top 5 results are relevant
#         MRR — where does the first relevant result appear
#
#
# INPUT FILES:
#   - chroma_db/                        ChromaDB from vector_db.py
#   - outputs/embeddings_meta.json      model name used to encode queries
#   - outputs/metadata_normalised.json  normalised popularity signals
#   - outputs/eval_queries.json         manually annotated evaluation set
#
# ============================================================

import json
import random
import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from reranker import load_reranker, retrieve_candidates, rerank
from metadata_fusion import load_metadata, fuse

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# CONFIGURATION

CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "prompts"
EMBEDDINGS_PATH  = "outputs/embeddings_meta.json"
EVAL_PATH = "eval_queries.json"
BEST_WEIGHTS_PATH = "outputs/best_weights.json"

# best weights found by Bayesian Optimisation in metadata_fusion.py
with open(BEST_WEIGHTS_PATH, encoding="utf-8") as f:
    best_weights = json.load(f)

BEST_ALPHA = best_weights["alpha"]
BEST_BETA  = best_weights["beta"]
BEST_GAMMA = best_weights["gamma"]
BEST_DELTA = best_weights["delta"]

# how many candidates to retrieve from ChromaDB
DEFAULT_CANDIDATES = 50



# METRICS
def precision_at_k(results, relevant_ids, k=5):
    """
    Precision@K — what fraction of the top K results are relevant.
    """
    top_k_ids = [r["id"] for r in results[:k]]
    relevant_found = sum(1 for pid in top_k_ids if pid in relevant_ids)
    return relevant_found / k


def mean_reciprocal_rank(results, relevant_ids):
    """
    MRR — score based on where the FIRST relevant result appears.
    """
    for rank, result in enumerate(results[:10], start=1):
        if result["id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


# EVALUATE ONE CONFIGURATION
def evaluate_configuration(results_per_query, eval_queries, label):
    """
    Compute Precision@5 and MRR for one pipeline configuration.
    Returns a dict with the computed scores for this configuration.
    """
    precision_scores = []
    mrr_scores = []

    for i, item in enumerate(eval_queries):
        relevant_ids = set(item["relevant_ids"])
        results = results_per_query[i]

        precision_scores.append(precision_at_k(results, relevant_ids, k=5))
        mrr_scores.append(mean_reciprocal_rank(results, relevant_ids))

    mean_precision = round(sum(precision_scores) / len(precision_scores), 4)
    mean_mrr = round(sum(mrr_scores) / len(mrr_scores), 4)

    return {
        "label": label,
        "precision": mean_precision,
        "mrr": mean_mrr,
    }


# RUN ALL THREE CONFIGURATIONS
def run_all_configurations(collection, encoder, reranker_model, metadata_lookup, eval_queries):
    """
    Run the full evaluation set through all three pipeline configurations
    and return results for each so they can be compared.
    """

    # store results for each configuration
    results_vector = []   # configuration A: vector search only
    results_reranker = []   # configuration B: vector + reranker
    results_fused = []   # configuration C: full pipeline

    print("Running evaluation over", len(eval_queries), "queries...")

    for i, item in enumerate(eval_queries):
        query_text = item["query"]
        print("Query", i + 1, "/", len(eval_queries), ":", query_text)

        # STEP 1 — retrieve candidates from ChromaDB (same for all configs)
        candidates = retrieve_candidates(collection, encoder, query_text)

        # CONFIGURATION A — vector search only
        # candidates are already sorted by cosine similarity, use as-is
        results_vector.append(candidates)

        # CONFIGURATION B — vector search + reranker
        reranked = rerank(reranker_model, query_text, candidates)
        results_reranker.append(reranked)

        # CONFIGURATION C — full pipeline with metadata fusion
        # use the best weights found by Bayesian Optimisation
        fused = fuse(reranked, metadata_lookup,
                     alpha=BEST_ALPHA,
                     beta=BEST_BETA,
                     gamma=BEST_GAMMA,
                     delta=BEST_DELTA)
        results_fused.append(fused)

    return results_vector, results_reranker, results_fused



# PRINT COMPARISON TABLE
def print_comparison_table(scores):
    """
    Print a clean comparison table of all three configurations.
    """

    print("EVALUATION RESULTS")
    print(f"{'Configuration':<30}  {'P@5':>6}  {'MRR':>6}")
    print("-" * 55)
    for s in scores:
        print(f"{s['label']:<30}  {s['precision']:>6}  {s['mrr']:>6}")
    print("=" * 55)

    # show improvement from A to C
    p_improvement = round(scores[2]["precision"] - scores[0]["precision"], 4)
    mrr_improvement = round(scores[2]["mrr"] - scores[0]["mrr"], 4)
    print("Overall improvement (A to C):")
    print("  Precision@5:", "+" + str(p_improvement) if p_improvement >= 0 else str(p_improvement))
    print("  MRR:        ", "+" + str(mrr_improvement) if mrr_improvement >= 0 else str(mrr_improvement))


# MAIN
def main():
    print("EVALUATION")

    # load the annotated evaluation set
    with open(EVAL_PATH, encoding="utf-8") as f:
        all_queries = json.load(f)
    eval_queries = all_queries["test"]
    print("Loaded", len(eval_queries), "test queries")

    # best weights are frozen
    print("Using weights:")
    print("alpha =", BEST_ALPHA)
    print("beta  =", BEST_BETA)
    print("gamma =", BEST_GAMMA)
    print("delta =", BEST_DELTA)

    # load metadata lookup
    metadata_lookup = load_metadata()

    # connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # load the query encoder
    with open(EMBEDDINGS_PATH, encoding="utf-8") as f:
        embedding_meta = json.load(f)
    encoder = SentenceTransformer(embedding_meta["model_name"])

    # load the reranker
    reranker_model = load_reranker()

    # run all three configurations over the full evaluation set
    results_vector, results_reranker, results_fused = run_all_configurations(
        collection, encoder, reranker_model, metadata_lookup, eval_queries
    )

    # compute metrics for each configuration
    score_a = evaluate_configuration(results_vector, eval_queries, "A) Vector search only")
    score_b = evaluate_configuration(results_reranker, eval_queries, "B) Vector + reranker")
    score_c = evaluate_configuration(results_fused, eval_queries, "C) Full pipeline (+ fusion)")

    # print the comparison table
    print_comparison_table([score_a, score_b, score_c])

    print("\nEvaluation done.")


if __name__ == "__main__":
    main()
