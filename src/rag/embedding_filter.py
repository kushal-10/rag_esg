import pandas as pd
import glob
import json
import os
from tqdm import tqdm

THRESHOLD = 0.5

# Find all sentence_scores.csv files
csv_paths = glob.glob("data/textsv2/*/*/sentence_scores.csv")

for csv_path in tqdm(csv_paths):
    dir_path = os.path.dirname(csv_path)
    splits_path = os.path.join(dir_path, "splits.json")
    embedding_filter_path = os.path.join(dir_path, "embedding_filter.json")

    # Read splits.json
    if not os.path.exists(splits_path):
        print(f"Warning: {splits_path} missing, skipping.")
        continue
    with open(splits_path, "r") as f:
        splits = json.load(f)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Auto-detect AI/SDG columns (edit if your names are different)
    ai_fuzzy_cols = [col for col in df.columns if "Fuzzy" in col or col.lower().startswith("ai_fuzzy")]
    sdg_cols = [col for col in df.columns if col not in ["chunk_id"] + ai_fuzzy_cols]

    # Keep only numeric columns in sdg_cols (e.g., drop sentence text if present)
    sdg_cols = [col for col in sdg_cols if pd.api.types.is_numeric_dtype(df[col])]

    # Select rows with any AI/SDG score above threshold
    mask = (df[ai_fuzzy_cols + sdg_cols] > THRESHOLD).any(axis=1)
    filtered = df.loc[mask, "chunk_id"].astype(str)

    # Prepare output
    output = [
        {"chunk_id": chunk_id, "sentence": splits.get(chunk_id, "")}
        for chunk_id in filtered
    ]

    # Write JSON
    with open(embedding_filter_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

print("Done! Each company/year now has an embedding_filter.json file.")
