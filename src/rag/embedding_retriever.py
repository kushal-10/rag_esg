"""Embed sentences and compute similarity scores against keyword sets."""

import json
import os
from typing import List

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ========== CONFIG ==========
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast & light
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 128  # Increase if you have spare RAM

KEYWORDS_EXTRA = [
    "Artificial Intelligence",
    "Machine Learning",
    "Reinforcement Learning",
    "Deep Learning",
    "Computer Vision",
    "Natural Language Processing",
]

# ========== MAIN LOGIC ==========


def get_keywords(sdgs_path: str) -> List[str]:
    """Load SDG keywords and append additional AI terms.

    Args:
        sdgs_path: Path to ``sdgs.json`` containing SDG mappings.

    Returns:
        List[str]: Combined list of SDG and extra AI keywords.
    """

    with open(sdgs_path, "r") as f:
        sdgs_data = json.load(f)
    return list(sdgs_data.values()) + KEYWORDS_EXTRA


def embed_texts(
    model: SentenceTransformer,
    sentences: List[str],
    batch_size: int = BATCH_SIZE,
):
    """Embed sentences using the provided SentenceTransformer model.

    Args:
        model: Loaded SentenceTransformer model.
        sentences: List of sentences to embed.
        batch_size: Batch size for encoding.

    Returns:
        numpy.ndarray: Normalized sentence embeddings.
    """

    return model.encode(
        sentences,
        batch_size=batch_size,
        device=DEVICE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def assign_scores(
    json_path: str,
    keywords: List[str],
    key_embeds,
    model: SentenceTransformer,
) -> None:
    """Compute similarity scores between sentence chunks and keywords.

    Args:
        json_path: Path to ``splits.json`` containing sentence chunks.
        keywords: List of keywords to score against.
        key_embeds: Embeddings of the keywords.
        model: Loaded SentenceTransformer model.
    """

    with open(json_path, "r") as f:
        chunks = json.load(f)

    if chunks:
        chunk_ids = list(chunks.keys())
        sentences = list(chunks.values())

        # Embed all at once
        sent_embeds = embed_texts(model, sentences)

        # Compute cosine similarities (batched)
        scores = (sent_embeds @ key_embeds.T).tolist()  # shape: [num_sentences, num_keywords]

        # Build dataframe
        df = pd.DataFrame(scores, columns=keywords)
        df.insert(0, "chunk_id", chunk_ids)

        # Save
        save_path = json_path.replace("splits.json", "sentence_scores.csv")
        df.to_csv(save_path, index=False)
    else:  # pragma: no cover - diagnostic output
        print(f"No chunks found! Skipping - {json_path}...")


if __name__ == "__main__":
    sdgs_path = os.path.join("data", "sdgs.json")
    keywords = get_keywords(sdgs_path)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    key_embeds = embed_texts(model, keywords)
    # Find files
    json_files: List[str] = []
    for dirname, _, filenames in os.walk(os.path.join("data", "textsv2")):
        for filename in filenames:
            if filename.endswith("splits.json"):
                json_files.append(os.path.join(dirname, filename))

    # Process
    for json_file in tqdm(json_files):
        save_path = json_file.replace("splits.json", "sentence_scores.csv")
        if not os.path.exists(save_path):
            assign_scores(json_file, keywords, key_embeds, model)
