# ============================================================
# FILE: metadata_fusion.py
# ROLE: Phase 4b — Combine the reranker relevance score with
#       popularity signals (likes, upvotes, uses) from the
#       dataset metadata to produce a final ranking score.
#
#       The formula is a weighted linear combination:
#       final_score = α × reranker_score + β × norm_likes
#                   + γ × norm_upvotes   + δ × norm_uses
#
#       The weights α, β, γ, δ are tuned using
#       Bayesian Optimisation (Optuna with TPE sampler).
#
# INPUT FILES:
#   - chroma_db/                        ChromaDB from vector_db.py
#   - outputs/embeddings_meta.json      model name used to encode queries
#   - outputs/metadata_normalised.json  normalised popularity signals from preprocessing.py
#
# OUTPUT FUNCTIONS:
#   - load_metadata()                   loads metadata_normalised.json into a dict
#   - fuse()                            returns reranked candidates re-sorted by final_score
#   - tune_weights()                    uses Bayesian Optimisation to find best α, β, γ, δ
#
# ============================================================

import json
import random
import numpy as np
import torch
import chromadb
import optuna
from sentence_transformers import SentenceTransformer
from reranker import load_reranker, rerank

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# silence optuna's per-trial log messages so output stays readable
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

CHROMA_DB_PATH       = "./chroma_db"
COLLECTION_NAME      = "prompts"
EMBEDDINGS_PATH      = "outputs/embeddings_meta.json"
METADATA_PATH        = "outputs/metadata_normalised.json"

# how many candidates to retrieve from ChromaDB before reranking
DEFAULT_CANDIDATES   = 50

# default weights — used when running without Bayesian Optimisation.
# α is high because the reranker score is the strongest signal.
# β, γ, δ are small boosts from popularity signals.
DEFAULT_ALPHA = 0.7   # weight for reranker relevance score
DEFAULT_BETA  = 0.1   # weight for normalised likes
DEFAULT_GAMMA = 0.1   # weight for normalised upvotes
DEFAULT_DELTA = 0.1   # weight for normalised uses

# how many Bayesian Optimisation trials to run when tuning weights.
# each trial = one evaluation of the objective function.
# 50 is enough to find a good combination without taking too long.
N_TRIALS = 50


# ---------------------------------------------------------------
# LOAD METADATA
# ---------------------------------------------------------------

def load_metadata(path=METADATA_PATH):
    """
    Load metadata_normalised.json and return it as a dictionary
    where the key is the prompt ID and the value is the metadata dict.

    This makes lookups fast — instead of looping through 18,709
    records every time, we just do metadata_lookup["pk_03138"].

    The metadata contains normalised (0-1) popularity signals:
    likes, upvotes, views, uses, author_reputation, net_score.
    These were computed and normalised by Person 1 in preprocessing.py.
    """
    with open(path) as f:
        records = json.load(f)

    # build a dict: { "pk_03138": { "likes": 0.096, "upvotes": 0.092, ... }, ... }
    metadata_lookup = {}
    for record in records:
        metadata_lookup[record["id"]] = record

    print("Loaded metadata for", len(metadata_lookup), "prompts")
    return metadata_lookup


# ---------------------------------------------------------------
# RETRIEVE CANDIDATES FROM CHROMADB
# ---------------------------------------------------------------

def retrieve_candidates(collection, encoder, query_text, n=DEFAULT_CANDIDATES):
    """
    Embed the query and retrieve the N closest prompts from ChromaDB
    by cosine similarity. Returns a list of candidate dicts.

    This is the same retrieval logic as in reranker.py — it lives
    here too so metadata_fusion.py can run the full pipeline
    independently without calling reranker.run_sample_queries().
    """

    # turn the query string into a vector
    query_vector = encoder.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    # find the N closest prompts in ChromaDB by cosine similarity
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    # unpack ChromaDB response into a list of candidate dicts
    candidates = []
    ids        = raw["ids"][0]
    documents  = raw["documents"][0]
    metadatas  = raw["metadatas"][0]
    distances  = raw["distances"][0]

    rank = 1
    for i in range(len(ids)):
        candidate = {
            "id":         ids[i],
            "content":    documents[i],
            "metadata":   metadatas[i],
            "similarity": 1.0 - float(distances[i]),
            "rank":       rank,
        }
        candidates.append(candidate)
        rank = rank + 1

    return candidates


# ---------------------------------------------------------------
# FUSE
# ---------------------------------------------------------------

def fuse(reranked, metadata_lookup, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
         gamma=DEFAULT_GAMMA, delta=DEFAULT_DELTA):
    """
    Combine reranker score with popularity signals into a final score.

    reranked         : list[dict] — output of rerank() from reranker.py,
                                    sorted by reranker_score, each dict
                                    has "id" and "reranker_score"
    metadata_lookup  : dict       — output of load_metadata(), keyed by prompt id
    alpha            : float      — weight for reranker score  (semantic relevance)
    beta             : float      — weight for normalised likes (popularity)
    gamma            : float      — weight for normalised upvotes (community approval)
    delta            : float      — weight for normalised uses (real usage signal)

    Returns the same list re-sorted by final_score, with two new keys added:
    "final_score" and "final_rank".

    WHY THIS FORMULA:
    The reranker score alone tells us how relevant a prompt is to the query.
    But two equally relevant prompts are not equally good — one might have
    thousands of uses and high likes, meaning real users found it valuable.
    The weighted combination lets us break ties using popularity signals.
    α is kept high (0.7) so relevance always dominates over popularity.

    LECTURE CONNECTION:
    α, β, γ, δ are hyperparameters of this scoring function.
    tune_weights() below uses Bayesian Optimisation to find the best values,
    exactly as shown in the course notebook with Optuna.
    """

    # the reranker scores are raw logits (roughly -10 to +10).
    # the metadata signals are already normalised to 0-1 by Person 1.
    # to make them comparable we normalise the reranker scores to 0-1 too.
    all_scores = [c["reranker_score"] for c in reranked]
    min_score  = min(all_scores)
    max_score  = max(all_scores)
    score_range = max_score - min_score

    for candidate in reranked:
        prompt_id = candidate["id"]

        # normalise reranker score to 0-1 range
        # if all scores are identical (score_range = 0) set to 1.0
        if score_range > 0:
            norm_reranker = (candidate["reranker_score"] - min_score) / score_range
        else:
            norm_reranker = 1.0

        # look up popularity signals for this prompt.
        # if a prompt is missing from metadata (shouldn't happen but just in case)
        # default all signals to 0.
        meta = metadata_lookup.get(prompt_id, {})
        norm_likes    = meta.get("likes",    0.0)
        norm_upvotes  = meta.get("upvotes",  0.0)
        norm_uses     = meta.get("uses",     0.0)

        # weighted linear combination — the core formula
        candidate["final_score"] = (
            alpha * norm_reranker +
            beta  * norm_likes   +
            gamma * norm_upvotes +
            delta * norm_uses
        )

    # sort by final score, highest first
    fused = sorted(reranked, key=lambda c: c["final_score"], reverse=True)

    # assign final rank numbers starting from 1
    final_rank = 1
    for candidate in fused:
        candidate["final_rank"] = final_rank
        final_rank = final_rank + 1

    return fused


# ---------------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------------

def print_fused_results(results, top_n=5):
    """
    Print a table of the top N results after metadata fusion.
    Shows reranker rank, final rank, and final score so you can
    see how much the popularity signals shifted the ordering.
    """
    print(f"\n{'Final':>5}  {'Rerank':>6}  {'Score':>7}  {'ID':<10}  {'Category':<18}  Title")
    print("-" * 75)
    for r in results[:top_n]:
        final_rank   = r.get("final_rank",    "?")
        reranker_rank = r.get("reranker_rank", "?")
        score        = round(r.get("final_score", 0), 3)
        prompt_id    = r.get("id", "")
        category     = r.get("metadata", {}).get("category", "")[:18]
        title        = r.get("metadata", {}).get("title",    "")[:40]
        print(f"{final_rank:>5}  {reranker_rank:>6}  {score:>7}  {prompt_id:<10}  {category:<18}  {title}")


# ---------------------------------------------------------------
# TUNE WEIGHTS WITH BAYESIAN OPTIMISATION
# ---------------------------------------------------------------

def tune_weights(collection, encoder, reranker_model, metadata_lookup, eval_queries):
    """
    Use Bayesian Optimisation (Optuna, TPE sampler) to find the best
    values of alpha, beta, gamma, delta that maximise search quality.

    collection       : ChromaDB collection
    encoder          : SentenceTransformer — query encoder
    reranker_model   : CrossEncoder — loaded reranker from reranker.py
    metadata_lookup  : dict — output of load_metadata()
    eval_queries     : list[dict] — each dict has:
                         "query"      : str        — the search query
                         "relevant_ids": list[str] — IDs of relevant prompts

    HOW BAYESIAN OPTIMISATION WORKS HERE (from lecture):
    Instead of trying every combination of weights (grid search),
    Optuna builds a probabilistic model (TPE) of how each weight
    combination affects the score. It uses this model to pick the
    most promising combination to try next. After N_TRIALS attempts
    it returns the best weights found.

    Each "trial" in Optuna is one run of the objective function below,
    with a specific set of weights suggested by the TPE algorithm.
    The objective function returns Precision@5 — how many of the top 5
    results are actually relevant — which Optuna tries to maximise.
    """

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.5, 1.0)
        beta = trial.suggest_float("beta", 0.0, 0.3)
        gamma = trial.suggest_float("gamma", 0.0, 0.3)
        delta = trial.suggest_float("delta", 0.0, 0.3)

        precision_scores = []

        for item in eval_queries:
            query_text = item["query"]
            relevant_ids = set(item["relevant_ids"])

            candidates = retrieve_candidates(collection, encoder, query_text)
            reranked = rerank(reranker_model, query_text, candidates)
            fused = fuse(reranked, metadata_lookup, alpha, beta, gamma, delta)

            top5_ids = [c["id"] for c in fused[:5]]
            relevant_found = sum(1 for pid in top5_ids if pid in relevant_ids)
            precision = relevant_found / 5
            precision_scores.append(precision)

        mean_precision = sum(precision_scores) / len(precision_scores)

        print("  Trial", trial.number + 1, "/", N_TRIALS, "— Precision@5:", round(mean_precision, 4))

        return mean_precision

    # create an Optuna study — same pattern as in the lecture notebook:
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)
    print("\nRunning Bayesian Optimisation for", N_TRIALS, "trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best Precision@5:  ", round(study.best_value, 4))
    print("Best weights found:", study.best_params)

    return study.best_params


# ---------------------------------------------------------------
# SAMPLE QUERIES
# ---------------------------------------------------------------

def run_sample_queries(collection, encoder, reranker_model, metadata_lookup):
    """
    Run two example queries through the full pipeline:
    retrieve → rerank → fuse with default weights.
    Prints before (reranker) and after (fusion) tables.
    """

    print("\nSAMPLE QUERIES")

    sample_queries = [
        "summarise a long document",
        "write a job application cover letter",
    ]

    for query_text in sample_queries:
        print("\nQuery:", query_text)

        # retrieve 50 candidates from ChromaDB
        candidates = retrieve_candidates(collection, encoder, query_text)

        # rerank with cross-encoder
        reranked = rerank(reranker_model, query_text, candidates)

        print("\nAFTER reranking (top 5):")
        # reuse reranker's print function for the reranker-only view
        from reranker import print_reranked_results
        print_reranked_results(reranked, top_n=5)

        # fuse with metadata using default weights
        fused = fuse(reranked, metadata_lookup)

        print("\nAFTER metadata fusion (top 5):")
        print_fused_results(fused, top_n=5)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    print("METADATA FUSION")

    # load metadata lookup dict from preprocessing.py output
    metadata_lookup = load_metadata()

    # connect to ChromaDB
    client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # load the same query encoder used in vector_db.py and reranker.py
    with open(EMBEDDINGS_PATH) as f:
        embedding_meta = json.load(f)
    encoder = SentenceTransformer(embedding_meta["model_name"])

    # load the cross-encoder reranker from reranker.py
    reranker_model = load_reranker()

    # run sample queries with default weights to verify the pipeline works
    run_sample_queries(collection, encoder, reranker_model, metadata_lookup)

    # ---------------------------------------------------------------
    # BAYESIAN OPTIMISATION
    # To tune weights you need an evaluation set — a list of queries
    # with known relevant prompt IDs, provided by Person 1.
    # Replace the example below with the real evaluation set once
    # Person 1 shares it.
    # ---------------------------------------------------------------

    # examples from personally built evaluation set
    with open("outputs/eval_queries.json") as f:
        eval_queries = json.load(f)

    best_weights = tune_weights(
        collection, encoder, reranker_model, metadata_lookup, eval_queries
    )
    print("\nUse these weights in fuse() for your final evaluation:")
    print("alpha =", best_weights["alpha"])
    print("beta  =", best_weights["beta"])
    print("gamma =", best_weights["gamma"])
    print("delta =", best_weights["delta"])

    # save best weights to file so evaluation.py can load them
    with open("outputs/best_weights.json", "w") as f:
        json.dump(best_weights, f, indent=2)
    print("Saved best weights to outputs/best_weights.json")

    print("\nMetadata fusion done. Import fuse() and load_metadata() from this file")
    print("in evaluation.py.")


if __name__ == "__main__":
    main()
