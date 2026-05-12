# ============================================================
# FILE: visualise_embeddings.py
# ROLE: Visualise prompt embeddings in 2D using UMAP and t-SNE,
#       colored by category. Only top 15 categories shown.
#
# INPUT FILES:
#   - outputs/embeddings.npy
#   - outputs/embeddings_ids.json
#   - outputs/cleaned_data.json
#
# OUTPUT FILES:
#   - images/umap_by_category.png
#   - images/tsne_by_category.png
# ============================================================

import json
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# CONFIGURATION
EMBEDDINGS_PATH = "outputs/embeddings.npy"
IDS_PATH        = "outputs/embeddings_ids.json"
CLEANED_PATH    = "outputs/cleaned_data.json"
IMAGES_DIR      = "images"

TOP_N_CATS  = 15
PER_CAT     = 200    # 200 per category = 3000 points total, t-SNE gets slow above that


# LOAD AND SAMPLE
def load_and_sample():
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    print("Embeddings:", embeddings.shape)

    with open(IDS_PATH, encoding="utf-8") as f:
        ids = json.load(f)

    with open(CLEANED_PATH, encoding="utf-8") as f:
        records = json.load(f)

    id_to_cat = {r["id"]: r.get("category", "unknown") for r in records}
    all_cats  = [id_to_cat.get(pid, "unknown") for pid in ids]

    print("Total unique categories in dataset:", len(set(all_cats)))

    # sample PER_CAT points from each of the top 15 categories
    counts = Counter(all_cats)
    sample_idx  = []
    sample_cats = []

    for cat, _ in counts.most_common(TOP_N_CATS):
        idx_for_cat = [i for i, c in enumerate(all_cats) if c == cat]
        chosen = random.sample(idx_for_cat, min(PER_CAT, len(idx_for_cat)))
        sample_idx.extend(chosen)
        sample_cats.extend([cat] * len(chosen))

    print("Categories:", [cat for cat, _ in counts.most_common(TOP_N_CATS)])
    print("Sampled", len(sample_cats), "prompts across", TOP_N_CATS, "categories")
    return embeddings[sample_idx], sample_cats


# PLOT
def save_plot(projected, sample_cats, cat_to_color, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(13, 8))

    for cat, color in cat_to_color.items():
        mask = [i for i, c in enumerate(sample_cats) if c == cat]
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=[color] * len(mask),
            s=10,
            alpha=0.65,
            linewidths=0,
            label=cat,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, title_fontsize=10, markerscale=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


# MAIN
def main():
    X, sample_cats = load_and_sample()

    # one color per category, tab20 has 20 distinct colors
    cmap = plt.colormaps["tab20"]
    cat_to_color = {cat: cmap(i / 20) for i, cat in enumerate(dict.fromkeys(sample_cats))}

    # UMAP
    X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(X)
    save_plot(X_umap, sample_cats, cat_to_color,
              title=f"UMAP — prompt embeddings by category (top {TOP_N_CATS}, n={len(sample_cats)})",
              xlabel="UMAP 1", ylabel="UMAP 2",
              path=f"{IMAGES_DIR}/umap_by_category.png")

    # t-SNE — PCA to 50 dims first
    pca   = PCA(n_components=50, random_state=SEED)
    X_pca = pca.fit_transform(X)

    print("Explained variance", round(pca.explained_variance_ratio_.sum() * 100, 1), "%")

    X_tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                  init="random", learning_rate="auto", random_state=SEED).fit_transform(X_pca)
    save_plot(X_tsne, sample_cats, cat_to_color,
              title=f"t-SNE — prompt embeddings by category (top {TOP_N_CATS}, n={len(sample_cats)})",
              xlabel="t-SNE dim 1", ylabel="t-SNE dim 2",
              path=f"{IMAGES_DIR}/tsne_by_category.png")


if __name__ == "__main__":
    main()