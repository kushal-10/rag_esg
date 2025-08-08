"""Simple sentiment classification with DistilBERT."""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load pre-trained tokenizer and model once at module import
TOKENIZER = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
MODEL = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


def classify_sentiment(text: str) -> str:
    """Classify the sentiment of ``text`` using DistilBERT.

    Args:
        text: Input string to analyse.

    Returns:
        str: Predicted sentiment label (e.g., ``POSITIVE`` or ``NEGATIVE``).
    """

    inputs = TOKENIZER(text, return_tensors="pt")
    with torch.no_grad():
        logits = MODEL(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return MODEL.config.id2label[predicted_class_id]


if __name__ == "__main__":
    print(classify_sentiment("neutral"))
