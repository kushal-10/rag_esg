""" Util functions for Filtering"""
import os

from langdetect import detect, DetectorFactory

from src.utils.file_utils import load_json, save_json
from sentence_transformers import SentenceTransformer

DetectorFactory.seed = 0  # Ensures consistent results

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# embeddings = model.encode(sentences)

def get_embeddings_ai(terms):
    embedding_data = {}
    for term in terms:
        embedding = model.encode(term)
        embedding_data[term] = embedding
    return embedding_data

def get_embeddings_sdgs(sdg_data):
    embedding_data = {}
    for k,v in sdg_data.items():
        embedding_data[k] = model.encode(v)
    return embedding_data

ai_terms = [
    "Artificial Intelligence",
    "Machine Learning",
    "Reinforcement Learning",
    "Deep Learning",
    "Computer Vision",
    "Natural Language Processing",
]
ai_embeddings = get_embeddings_ai(ai_terms)

ai_terms_de = [
    "Künstliche Intelligenz",       # Artificial Intelligence
    "Maschinelles Lernen",          # Machine Learning
    "Bestärkendes Lernen",          # Reinforcement Learning
    "Tiefes Lernen",                # Deep Learning
    "Computer Vision",              # Computer Vision (often used untranslated)
    "Natürliche Sprachverarbeitung" # Natural Language Processing
]
ai_embeddings_de = get_embeddings_ai(ai_terms_de)

sdgs = load_json(os.path.join("data", "sdgs.json"))
sdg_embeddings = get_embeddings_sdgs(sdgs)

sdgs_de = load_json(os.path.join("data", "sdgs_de.json"))
sdgs_embeddings_de = get_embeddings_sdgs(sdgs_de)


def detect_german(text: str) -> bool:
    """
        Detects if the input text is in German ('de') by analyzing
        only the first 10,000 characters.

        Args:
            text (str): The text to analyze.

        Returns:
            bool: True if the language is German, False otherwise.
        """
    try:
        truncated_text = text[:10000]
        language = detect(truncated_text)
        return language == 'de'
    except:
        return False
