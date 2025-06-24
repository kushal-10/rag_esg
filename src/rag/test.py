"""
Test out different text splitters on a sample text file
"""
import re
import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import pandas as pd

def recursive_splitter(text_path: str):
    """
    Basic paragraph based splitter
    """

    # Load the text
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Normalize line breaks (some PDFs have \r\n, some \n)
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split into paragraphs — assume paragraphs are separated by 1 or more blank lines
    paragraphs = re.split(r'\n\s*\n', text)

    # Clean empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs

def clean_pdf_text(raw_text):
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

    text = re.sub(r'Page\s+\d+', '', text)  # Remove page numbers

    text = re.sub(r'-\n', '', text)  # Fix hyphenation

    text = re.sub(r'\n([a-z])', r' \1', text)  # Merge broken lines in paragraph

    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks

    text = re.sub(r'[ \t]+', ' ', text)

    text = text.strip()

    return text

def semantic_splitter(text_path: str):
    """
    Semantic based splitter
    """

    # Load the text
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_splitter = SemanticChunker(OpenAIEmbeddings())

    docs = text_splitter.create_documents([text])

    return docs


if __name__ == "__main__":
    text_path = os.path.join("data", "texts", "1.sap_$240.94 b_information tech", "2014", "results.txt")

    paragraphs = recursive_splitter(text_path)
    # Show the paragraphs
    lengths = []
    for i, p in enumerate(paragraphs):
        # print(f'--- Paragraph {i + 1} ---')
        # print(p)
        p = clean_pdf_text(p)
        words = p.split(" ")
        lengths.append(len(words))
        # print('\n')
    print(max(lengths), min(lengths), sum(lengths) / len(lengths))
    print(len(lengths))

    # docs = semantic_splitter(text_path)
    # # Show the paragraphs
    # doc_lengths = []
    # for doc in docs:
    #     p = doc.page_content
    #     p = clean_pdf_text(p)
    #     words = p.split(" ")
    #     doc_lengths.append(len(words))
    #
    # print(max(doc_lengths), min(doc_lengths), sum(doc_lengths) / len(doc_lengths))




