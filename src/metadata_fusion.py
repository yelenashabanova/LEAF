# ============================================================
# FILE: metadata_fusion.py
# ROLE: Phase 4b — Combine the reranker relevance score with
#       popularity signals from dataset metadata to produce
#       a final ranking score using a weighted linear combination:
#       final_score = alpha × reranker_score + beta × norm_likes
#                   + gamma × norm_upvotes + delta × norm_uses
#       Weights are tuned via Bayesian Optimisation (Optuna, TPE).
#
# INPUT FILES:
#   - chroma_db/                        ChromaDB from vector_db.py
#   - outputs/embeddings_meta.json      query encoder model name
#   - outputs/metadata_normalised.json  normalised popularity signals
#   - outputs/eval_queries.json         annotated evaluation set
#
# OUTPUT FILES:
#   - outputs/best_weights.json         optimal weights from Bayesian Optimisation
#
# OUTPUT FUNCTIONS (used by evaluation.py):
#   - load_metadata()                   loads metadata into a lookup dict
#   - retrieve_candidates()             queries ChromaDB for top 50 candidates
#   - fuse()                            re-sorts reranked candidates by final_score
# ============================================================

import json
import random
import numpy as np
import torch
import chromadb
import optuna
from sentence_transformers import SentenceTransformer
from reranker import load_reranker, retrieve_candidates, rerank, print_reranked_results

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# silence optuna's per-trial log messages so output stays readable
optuna.logging.set_verbosity(optuna.logging.WARNING)


# CONFIGURATION

CHROMA_DB_PATH       = "./chroma_db"
COLLECTION_NAME      = "prompts"
EMBEDDINGS_PATH      = "outputs/embeddings_meta.json"
METADATA_PATH        = "outputs/metadata_normalised.json"

# number candidates to retrieve from ChromaDB before reranking
DEFAULT_CANDIDATES   = 50

# Bayesian Optimisation trials to run when tuning weights
N_TRIALS = 30



# LOAD METADATA
def load_metadata(path=METADATA_PATH):
    """
    Load metadata_normalised.json and return it as a lookup dictionary
    where the key is the prompt ID and the value is the metadata dict,
    including likes, upvotes, uses and so on.
    """
    with open(path) as f:
        records = json.load(f)

    # build a dict
    metadata_lookup = {}
    for record in records:
        metadata_lookup[record["id"]] = record

    print("Loaded metadata for", len(metadata_lookup), "prompts")
    return metadata_lookup


# TUNE WEIGHTS WITH BAYESIAN OPTIMISATION
def tune_weights(collection, encoder, reranker_model, metadata_lookup, eval_queries):
    """
    Use Bayesian Optimisation (Optuna, TPE sampler) to find the best
    values of alpha, beta, gamma, delta that maximise search quality.

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

    return study.best_params, study



# FUSE
def fuse(reranked, metadata_lookup, alpha, beta, gamma, delta):
    """
    Combine reranker score with popularity signals into a final score.
    The reranker score alone tells us how relevant a prompt is to the query.
    But two equally relevant prompts are not equally good — one might have
    thousands of uses and high likes, meaning real users found it valuable.

    Returns the same list re-sorted by final_score, with two new keys added:
    "final_score" and "final_rank".
    """

    # the reranker scores are raw logits
    # the metadata signals are already normalised
    # to make them comparable we normalise the reranker scores to 0-1 too
    all_scores = [c["reranker_score"] for c in reranked]
    min_score  = min(all_scores)
    max_score  = max(all_scores)
    score_range = max_score - min_score

    for candidate in reranked:
        prompt_id = candidate["id"]

        # normalise
        if score_range > 0:
            norm_reranker = (candidate["reranker_score"] - min_score) / score_range
        else:
            norm_reranker = 1.0

        # look up popularity signals for this prompt
        meta = metadata_lookup.get(prompt_id, {})
        norm_likes = meta.get("likes",    0.0)
        norm_upvotes = meta.get("upvotes",  0.0)
        norm_uses = meta.get("uses",     0.0)

        # weighted linear combination formula
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



# PRINT RESULTS
def print_fused_results(results, top_n=5):
    """
    Print a table of the top N results after metadata fusion.
    Shows reranker rank, final rank, and final score to see how much
    the popularity signals shifted the ordering.
    """
    print(f"\n{'Final':>5}  {'Rerank':>6}  {'Score':>7}  {'ID':<10}  {'Category':<18}  Title")
    print("-" * 75)
    for r in results[:top_n]:
        final_rank = r.get("final_rank",    "?")
        reranker_rank = r.get("reranker_rank", "?")
        score = round(r.get("final_score", 0), 3)
        prompt_id = r.get("id", "")
        category = r.get("metadata", {}).get("category", "")[:18]
        title = r.get("metadata", {}).get("title",    "")[:40]
        print(f"{final_rank:>5}  {reranker_rank:>6}  {score:>7}  {prompt_id:<10}  {category:<18}  {title}")




# MAIN
def main():
    print("METADATA FUSION")

    # load metadata lookup dict
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

    # load evaluation set and tune weights
    with open("outputs/eval_queries.json") as f:
        eval_queries = json.load(f)
    # after eval set is extended  load evaluation set like this instead
    #with open("outputs/eval_queries.json") as f:
    #    all_queries = json.load(f)
    #eval_queries = all_queries["validation"]

    best_weights, study = tune_weights(
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

    # generate and save visualizations
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig2 = optuna.visualization.plot_param_importances(study)
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig4 = optuna.visualization.plot_slice(study)

    for fig, name in [
        (fig1, "optimization_history"),
        (fig2, "param_importances"),
        (fig3, "parallel_coordinate"),
        (fig4, "param_slice"),
    ]:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.write_image("images/" + name + ".png")

    print("Saved visualisations to images/")

    # run sample queries with weights to verify the pipeline works
    print("\nSAMPLE QUERIES")

    sample_queries = [
        "summarise a long document",
        "write a job application cover letter",
    ]

    for query_text in sample_queries:
        print("\nQuery:", query_text)

        # retrieve 50 candidates from ChromaDB using function from reranker,py
        candidates = retrieve_candidates(collection, encoder, query_text)

        # rerank with cross-encoder with function from reranker.py
        reranked = rerank(reranker_model, query_text, candidates)

        print("\nAFTER reranking (top 5):")
        # reranker.py print function for the table
        print_reranked_results(reranked, top_n=5)

        # fuse with metadata using best weights
        fused = fuse(reranked, metadata_lookup,
                     alpha=best_weights["alpha"],
                     beta=best_weights["beta"],
                     gamma=best_weights["gamma"],
                     delta=best_weights["delta"])

        print("\nAFTER metadata fusion (top 5):")
        print_fused_results(fused, top_n=5)

    print("\nMetadata fusion done. Import fuse() and load_metadata() from this file")
    print("in evaluation.py.")


if __name__ == "__main__":
    main()
