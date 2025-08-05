"""
Test out different text splitters on a sample text file
"""
import re
import os
from typing import Dict, List
import logging
import json

from tqdm import tqdm
import nltk
nltk.download('punkt')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO,
                    filename=os.path.join("src", "rag", "splitter.log"),
                    filemode='w')

def clean_pdf_text(raw_text):
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

    text = re.sub(r'Page\s+\d+', '', text)  # Remove page numbers

    text = re.sub(r'-\n', '', text)  # Fix hyphenation

    text = re.sub(r'\n([a-z])', r' \1', text)  # Merge broken lines in paragraph

    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks

    text = re.sub(r'[ \t]+', ' ', text)

    text = text.replace('\u00ad', '') # rem soft hyphens

    text = text.replace('\u2009', '') # thinspaces

    text = text.strip()

    return text

def sentence_splitter(text_path: str):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_pdf_text(text)
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def save_splits_df(data:List, path:str):

    json_data = {}
    for i in range(len(data)):
        assert type(data[i]) == str, f"Data split is not in correct format! Expected str, got {type(data[i])}"
        json_data[str(i)] = data[i]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    logging.info(f'Saved splits dataframe to {path}')

def split_texts(base_dir: str):
    txt_files = []
    for dirname, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('results.de.de.txt'):
                file_path = os.path.join(dirname, filename)
                txt_files.append(file_path)


    for file_path in tqdm(txt_files):
        save_path = file_path.replace('results.de.de.txt', 'splits_de.json')
        sentences = sentence_splitter(file_path)
        save_splits_df(sentences, save_path)

    logging.info(f'Saved all splits as dataframes in respective "splits.json"')



if __name__ == "__main__":
    BASE_DIR = "data/textsv3"
    split_texts(BASE_DIR)


