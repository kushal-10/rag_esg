"""Augment sentence scores with RapidFuzz matches for AI-related terms."""

import glob
import json
import os
from typing import List

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

# --- Parameters ---
ai_terms = [
    "Artificial Intelligence",
    "Machine Learning",
    "Reinforcement Learning",
    "Deep Learning",
    "Computer Vision",
    "Natural Language Processing",
]
ai_colnames = [f"{term} Fuzzy" for term in ai_terms]
fuzzy_threshold = 90  # Now stricter!


def add_fuzzy_scores(
    csv_files: List[str] = None, ai_terms_list: List[str] = None, threshold: int = fuzzy_threshold
) -> None:
    """Add fuzzy-match columns for AI terms to sentence score CSV files.

    Args:
        csv_files: List of ``sentence_scores.csv`` paths. If ``None`` a default
            glob pattern is used.
        ai_terms_list: Terms to fuzzy match against sentences. ``ai_terms`` used
            if not provided.
        threshold: Minimum RapidFuzz score to count as a match.
    """

    if csv_files is None:
        csv_files = glob.glob("data/texts/*/*/sentence_scores.csv")
    if ai_terms_list is None:
        ai_terms_list = ai_terms

    log_path = "src/rag/fuzzy.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for csv_path in tqdm(csv_files, desc="Processing company/year files"):
        # 1. Load CSV and matching splits.json
        df = pd.read_csv(csv_path)
        splits_path = csv_path.replace("sentence_scores.csv", "splits.json")
        with open(splits_path, "r") as f:
            splits = json.load(f)
        # Ensure chunk IDs are strings
        df["chunk_id"] = df["chunk_id"].astype(str)

        # 2. Get the sentence text for each row
        df["sentence"] = df["chunk_id"].map(splits)

        # 3. Add fuzzy columns for each AI term
        for term, colname in zip(ai_terms_list, [f"{t} Fuzzy" for t in ai_terms_list]):
            tqdm_desc = f"{csv_path.split('/')[-3:]}, {term} Fuzzy"
            col_matches = []
            for chunk_id, sent in zip(df["chunk_id"], tqdm(df["sentence"], desc=tqdm_desc, leave=False)):
                if sent is not None:
                    score = fuzz.partial_ratio(term.lower(), sent.lower())
                    match = 1 if score >= threshold else 0
                    col_matches.append(match)
                else:
                    col_matches.append(0)
            df[colname] = col_matches

        # 4. Save back to CSV (overwrite)
        df.drop(columns=["sentence"], inplace=True)
        df.to_csv(csv_path, index=False)

    print(
        f"All CSVs updated with fuzzy columns (threshold={threshold}).\nMatches logged to {log_path}"
    )


if __name__ == "__main__":
    add_fuzzy_scores()
