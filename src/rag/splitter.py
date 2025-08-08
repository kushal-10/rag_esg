"""Split cleaned PDF text into sentences and persist as JSON."""

import json
import logging
import os
import re
from typing import List

import nltk
from tqdm import tqdm

nltk.download("punkt")

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename=os.path.join("src", "rag", "splitter.log"),
    filemode="w",
)


def clean_pdf_text(raw_text: str) -> str:
    """Remove artifacts and normalise raw PDF text.

    Args:
        raw_text: Text extracted from a PDF.

    Returns:
        str: Cleaned text.
    """

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"Page\s+\d+", "", text)  # Remove page numbers
    text = re.sub(r"-\n", "", text)  # Fix hyphenation
    text = re.sub(r"\n([a-z])", r" \1", text)  # Merge broken lines in paragraph
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # Normalize paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("\u00ad", "")  # Remove soft hyphens
    text = text.replace("\u2009", "")  # Remove thin spaces
    text = text.strip()
    return text


def sentence_splitter(text_path: str) -> List[str]:
    """Split a text file into individual sentences.

    Args:
        text_path: Path to the text file to split.

    Returns:
        List[str]: List of sentences.
    """

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_pdf_text(text)
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def save_splits_df(data: List[str], path: str) -> None:
    """Persist sentence splits as a JSON mapping.

    Args:
        data: List of sentence strings.
        path: Destination path for ``splits.json``.
    """

    json_data = {}
    for i, sentence in enumerate(data):
        assert isinstance(sentence, str), (
            f"Data split is not in correct format! Expected str, got {type(sentence)}"
        )
        json_data[str(i)] = sentence

    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved splits dataframe to {path}")


def split_texts(base_dir: str) -> None:
    """Generate sentence splits for all text files under ``base_dir``.

    Args:
        base_dir: Base directory containing text files.
    """

    txt_files: List[str] = []
    for dirname, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith("results.de.de.txt"):
                file_path = os.path.join(dirname, filename)
                txt_files.append(file_path)

    for file_path in tqdm(txt_files):
        save_path = file_path.replace("results.de.de.txt", "splits_de.json")
        sentences = sentence_splitter(file_path)
        save_splits_df(sentences, save_path)

    logging.info('Saved all splits as dataframes in respective "splits.json"')



if __name__ == "__main__":
    BASE_DIR = "data/textsv3"
    split_texts(BASE_DIR)


