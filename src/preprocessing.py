# ============================================================
# FILE: preprocessing.py
# ROLE: Load, explore, and clean the prompt dataset.
#       Outputs cleaned_data.json and metadata_normalised.json.
# OUTPUT FILES:
#   - cleaned_data.json used by Person 2 (embeddings.py)
#   - metadata_normalised.json used by Person 3 (metadata_fusion.py)
#   - scaler.pkl saved for future use on new data
# ============================================================

# ------------------------------------------------------------------
# IMPORTS
# Standard library
# pandas, sklearn, joblib need to be in requirements.txt.
# ------------------------------------------------------------------
import json
import os
import random
import re
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Reproducibility — (same output every run)
random.seed(42)
np.random.seed(42)


# ==================================================================
# SECTION 1 — CONFIGURATION
#
# Central configuration for file paths and preprocessing constants.
# Keeping these values here makes it easier to update the project
# structure without changing the rest of the script.
# ==================================================================

#Input dataset provided
DATASET_PATH = "LEAF-promptkaban-dataset/dataset.json"

CLEANED_OUTPUT_PATH = "outputs/cleaned_data.json"
METADATA_OUTPUT_PATH = "outputs/metadata_normalised.json"
SCALER_OUTPUT_PATH = "outputs/scaler.pkl"

# Columns that will be normalised to 0-1 range for metadata scoring.
# Numeric metadata fields selected for normalisation.
# These columns represent popularity, usage, and author-quality signals
# that may later be combined with semantic similarity scores.
METADATA_COLS_TO_NORMALISE = [
    "likes",
    "upvotes",
    "views",
    "uses",
    "author_reputation",
]

# Pattern used to identify template variables such as {{company_name}}
# or {{target_audience}} inside prompt texts.
PLACEHOLDER_PATTERN = re.compile(r"\{\{.*?\}\}|\{\{\w+")


# ==================================================================
# SECTION 2 — LOAD DATA
#
# Load the raw JSON dataset and run basic structural checks before
# moving to EDA and preprocessing.
# ==================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load the LEAF PromptKaban dataset from a JSON file

    The expected input is a list of prompt records, where each record
    contains the fields described in the project guide. The function
    converts the JSON data into a pandas DataFrame and checks that the
    main expected columns are present.

    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Make sure dataset.json is in the same folder as this script."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    expected_rows = 20_000
    if df.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rowsm but loaded {df.shape[0]:,}"
            "Please check that the correct dataset file is being used"
        )


    expected_columns = {
        "id", "title", "content", "category", "subcategory", "tags",
        "likes", "upvotes", "downvotes", "views", "uses",
        "author_reputation", "difficulty", "created_at",
        "version", "fork_count", "has_placeholders", "placeholders",
        "language", "target_model",
    }
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"The dataset is missing the following expected columns: "
            f"{sorted(missing_columns)}"
        )

    print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {df.columns.tolist()}")

    return df


# ==================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)

# Inspect the raw dataset before applying any cleaning or feature
# construction. This step does not modify the DataFrame.
# ==================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Print the main exploratory statistics used to understand the dataset.

    The output summarises:
    - dataset shape and column types
    - missing values
    - duplicate IDs and duplicate prompt contents
    - category, difficulty, language, and placeholder distributions
    - content length statistics
    - numeric metadata ranges
    """

    print("\n BASIC SHAPE")
    print(f"  Rows × Columns: {df.shape}")

    print("\n COLUMN TYPES")
    print(df.dtypes)

    print("\n NULL VALUES (missing data per column) ")
    null_counts = df.isnull().sum()
    print(null_counts)
    print(f"  -> Total cells with missing data: {null_counts.sum()}")

    print("\n DUPLICATES")
    n_dup_ids = df.duplicated(subset=["id"]).sum()
    n_dup_content = df.duplicated(subset=["content"]).sum()
    print(f"  Duplicate IDs: {n_dup_ids}")
    print(f"  Duplicate content strings: {n_dup_content}")

    print("\n CATEGORY DISTRIBUTION")
    print(df["category"].value_counts().to_string())

    print("\n CONTENT LENGTH (characters)")
    content_lengths = df["content"].str.len()
    print(content_lengths.describe().round(1))
    print(f"  -> Prompts with content shorter than 20 chars: {(content_lengths < 20).sum()}")

    print("\n DIFFICULTY DISTRIBUTION ")
    print(df["difficulty"].value_counts().to_string())

    print("\n PLACEHOLDER ")
    n_placeholders = df["has_placeholders"].sum()
    placeholder_share = n_placeholders / len(df) * 100
    print(f"  Prompts with placeholders: {n_placeholders:,} ({placeholder_share:.1f}%)")

    print("\n LANGUAGE DISTRIBUTION ")
    print(df["language"].value_counts().to_string())

    print("\n NUMERIC METADATA RANGES ")
    numeric_cols = ["likes", "upvotes", "downvotes", "views", "uses",
                    "author_reputation", "fork_count"]
    print(df[numeric_cols].describe().round(1).to_string())

    print("\n TOP TARGET MODELS ")
    print(df["target_model"].value_counts().head(10).to_string())

    print("\n EDA COMPLETE \n")


# ==================================================================
# SECTION 4 — HANDLE DUPLICATES AND NULLS
#
# Apply minimal cleaning before building the text used for embeddings.
# The goal is to remove records that would either break the pipeline
# or add little value to semantic search.
# ==================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset before embedding.

    The cleaning step removes duplicate IDs, rows without prompt content,
    very short prompt texts, and repeated content strings. It also fills
    missing numeric metadata values and standardises the tags column.
    """

    initial_count = len(df)
    print(f"  Starting rows: {initial_count:,}")

    # Remove duplicate prompt IDs, keeping the first occurrence.
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    print(
        f"  After duplicate ID removal: {len(df):,} rows "
        f"(dropped {before - len(df)})"
    )

    # Remove rows without content, since content is the core text used
    # for semantic embedding.
    before = len(df)
    df = df.dropna(subset=["content"])
    print(f"  After null content removal: {len(df):,} rows "
          f"(dropped {before - len(df)})")

    # Remove very short prompt texts. These records usually contain too
    # little semantic information to be useful for embedding-based search.
    before = len(df)
    df = df[df["content"].str.len() >= 20]
    print(f"  After short content removal (<20 chars): {len(df):,} rows "
          f"(dropped {before - len(df)})")

    # Remove exact duplicate prompt contents. Keeping repeated content
    # would create identical or near-identical embeddings in the vector DB.
    before = len(df)
    df = df.drop_duplicates(subset=["content"], keep="first")
    print(f"  After duplicate content removal: {len(df):,} rows "
          f"(dropped {before - len(df)})")

    # Fill missing numeric metadata with 0 so the normalisation step can
    # run without errors.
    for col in METADATA_COLS_TO_NORMALISE:
        df[col] = df[col].fillna(0)

    # Ensure tags are always stored as lists, because they will later be
    # joined into the text_to_embed field.
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])

    # Reset row index so it is continuous after dropping rows.

    df = df.reset_index(drop=True)

    print(f"\n  Total rows removed: {initial_count - len(df):,}")
    print(f"  Final clean dataset: {len(df):,} rows")

    return df


# ==================================================================
# SECTION 5 — HANDLE PLACEHOLDERS
#
# Replace template variables with a neutral token before embedding.
# This keeps the sentence structure while avoiding prompt-specific
# variable names such as {{company_name}} or {{recipient_name}}.
# ==================================================================

def handle_placeholders(text: str) -> str:
    """
    Replace {{variable}} template tokens with the neutral word [VALUE].

    Raw tokens like {{recipient_name}} are meaningless to an embedding model.
    Substituting [VALUE] preserves sentence structure while removing noise,
    so the model can still interpret the prompt's intent correctly.
    """
    cleaned = PLACEHOLDER_PATTERN.sub("[VALUE]", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()

# ==================================================================
# SECTION 6 — BUILD text_to_embed FIELD
#
# Constructs the string passed to the embedding model for each prompt.
# Combining title, content, and tags gives the model more context than
# content alone — the title provides a compact label, the tags provide
# topic keywords that may not appear explicitly in the body text.
# ==================================================================

def build_text_to_embed(row: pd.Series) -> str:
    """
    Concatenate title, content, and tags into a single string for embedding.

    Title goes first as a compact semantic label, followed by the full
    prompt body, then tags as topic keywords. A period separates the
    title from the content so the model treats them as distinct phrases.
    """
    title = str(row.get("title", "") or "")
    content = str(row.get("content", "") or "")

    tags = row.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags_str = " ".join(tags)

    return f"{title}. {content} {tags_str}".strip()


# ==================================================================
# SECTION 7 — NORMALISE METADATA
#
# Rescale numeric metadata columns to a common 0-1 range before they
# are used in metadata-based scoring.
# ==================================================================

def normalise_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalise selected metadata columns using MinMaxScaler.

    A net_score column is also created from likes, upvotes, and downvotes
    to provide a simple combined engagement signal.
    """

    meta = df[["id"] + METADATA_COLS_TO_NORMALISE].copy()

    # Combined engagement score before scaling.
    meta["net_score"] = df["likes"] + df["upvotes"] - df["downvotes"]

    cols_to_scale = METADATA_COLS_TO_NORMALISE + ["net_score"]

    scaler = MinMaxScaler()
    meta[cols_to_scale] = scaler.fit_transform(meta[cols_to_scale])

    mins = meta[cols_to_scale].min().round(4).to_dict()
    maxs = meta[cols_to_scale].max().round(4).to_dict()

    print(f"  Scaled mins: {mins}")
    print(f"  Scaled maxs: {maxs}")

    return meta, scaler

# ==================================================================
# SECTION 8 — SAVE OUTPUTS
#
# Writes the three output files that downstream team members depend on.
# Filenames are defined in Section 1, coordinate with the team before
# changing them, as Person 2 and Person 3 hardcode these paths.
# ==================================================================

def save_outputs(
        df_cleaned: pd.DataFrame,
        df_meta: pd.DataFrame,
        scaler: MinMaxScaler,
) -> None:
    """
    Save the preprocessing outputs.

    cleaned_data.json is used by Person 2 for the embedding phase.
    metadata_normalised.json is used by Person 3 for metadata fusion.
    scaler.pkl stores the fitted MinMaxScaler for future transformations.
    """

    # OUTPUT 1: cleaned dataset for Person 2
    records = df_cleaned.to_dict(orient="records")
    with open(CLEANED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(records):,} records -> {CLEANED_OUTPUT_PATH}")

    # OUTPUT 2: normalised metadata for Person 3
    meta_records = df_meta.to_dict(orient="records")
    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(meta_records):,} metadata rows -> {METADATA_OUTPUT_PATH}")

    # OUTPUT 3: fitted scaler for future use
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"  Saved scaler -> {SCALER_OUTPUT_PATH}")

# ==================================================================
# SECTION 9 — MAIN ORCHESTRATOR
#
# This is the function that ties everything together. It calls each
# step in the correct order. When you run `python preprocessing.py`
# this is what executes.
#
# Do NOT change the order of steps — each one depends on the previous.
# ==================================================================

def main():
    print("=" * 60)
    print("Phase 1: EDA & Preprocessing")
    print("=" * 60)

    # STEP 1: Load raw data
    print("\n[1/6] Loading dataset")
    df = load_data(DATASET_PATH)

    # STEP 2: Run EDA, prints stats, does not modify df
    print("\n[2/6] Running exploratory data analysis...")
    run_eda(df)

    # STEP 3: Clean,  remove duplicates, fill nulls
    print("\n[3/6] Cleaning data")
    df = clean_data(df)

    # STEP 4: Handle placeholders in the content field
    print("\n[4/6] Handling placeholders in content")
    df["content"] = df["content"].apply(handle_placeholders)

    # STEP 5: Build the text_to_embed field
    print("\n[5/6] Building text_to_embed field")
    df["text_to_embed"] = df.apply(build_text_to_embed, axis=1)
    # Sanity check, print one example
    print("Example text_to_embed (row 0):")
    print(df["text_to_embed"].iloc[0][:300])

    # STEP 6: Normalise metadata and save all outputs
    print("\n[6/6] Normalising metadata and saving outputs...")
    df_meta, scaler = normalise_metadata(df)
    save_outputs(df, df_meta, scaler)

    print("\n✓ Preprocessing complete.")
    print(f"  -> {CLEANED_OUTPUT_PATH}  (send to Person 2)")
    print(f"  -> {METADATA_OUTPUT_PATH} (send to Person 3)")
    print(f"  -> {SCALER_OUTPUT_PATH}")


# ==================================================================
# ENTRY POINT
# ==================================================================
if __name__ == "__main__":
    main()