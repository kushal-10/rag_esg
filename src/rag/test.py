"""
Test out different text splitters on a sample text file
"""
import re
import os
from typing import List
import nltk
nltk.download('punkt')

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

def sentence_splitter(text_path: str):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_pdf_text(text)
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

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

def save_df(doc_contents: List, path: str):
    df = {
        "data": doc_contents,
    }
    df = pd.DataFrame(df)
    df.to_csv(path, index=False)

def get_docs(text_path: str):
    paragraphs = recursive_splitter(text_path)
    # Show the paragraphs
    lengths = []
    page_contents = []
    for i, p in enumerate(paragraphs):
        # print(f'--- Paragraph {i + 1} ---')
        # print(p)
        p = clean_pdf_text(p)
        words = p.split(" ")
        lengths.append(len(words))
        page_contents.append(p)
        # print('\n')
    save_df(page_contents, os.path.join("results", "rec_split.csv"))

    docs = semantic_splitter(text_path)
    # Show the paragraphs
    doc_lengths = []
    page_contents = []
    for doc in docs:
        p = doc.page_content
        p = clean_pdf_text(p)
        page_contents.append(p)
        words = p.split(" ")
        doc_lengths.append(len(words))

    save_df(page_contents, os.path.join("results", "sem_split.csv"))

def analyse_docs(path: str):
    df = pd.read_csv(path)
    print("*"*50)
    print("Split 1")
    print(df['data'].iloc[0])
    print("*"*50)

    print("*" * 50)
    print("Split 2")
    print(df['data'].iloc[1])
    print("*" * 50)

    print("*" * 50)
    print("Split 3")
    print(df['data'].iloc[2])
    print("*" * 50)


if __name__ == "__main__":
    text_path = os.path.join("data", "texts", "1.sap_$240.94 b_information tech", "2014", "results.txt")
    # get_docs(text_path)

    sents_df = pd.read_csv(os.path.join("results", "sen_split.csv"))

    print(sents_df.head())

    counts = []
    for i in range(len(sents_df)):
        sent = sents_df["data"].iloc[i]
        words = sent.split(" ")
        counts.append(len(words))

    print(len(counts), sum(counts)/len(counts), max(counts), min(counts))
    """
    3995 28.39674593241552 603 1

    """


