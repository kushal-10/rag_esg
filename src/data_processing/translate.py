import json
import os
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from deep_translator import GoogleTranslator

# HuggingFace translation pipeline
translator = pipeline("translation", model="facebook/m2m100_418M")
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

def chunk_text(text, max_tokens=500):
    """Split text into chunks with at most max_tokens tokens each."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokens = tokenizer(" ".join(current_chunk), return_tensors="pt")["input_ids"]
        if tokens.shape[1] >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

root_dir = 'data/textsv3'

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith('splits_de.json'):
            file_path = os.path.join(root, file)
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                json_data = json.load(f)
            new_data = {}

            for k, v in tqdm(json_data.items()):
                try:
                    # Tokenize and check length
                    tokens = tokenizer(v, return_tensors="pt")["input_ids"]
                    if tokens.shape[1] > 500:
                        # Split into chunks and translate
                        chunks = chunk_text(v, max_tokens=500)
                        translated_chunks = []
                        for chunk in chunks:
                            trans = translator(chunk, src_lang="de", tgt_lang="en")[0]['translation_text']
                            translated_chunks.append(trans)
                        translated = " ".join(translated_chunks)
                    else:
                        translated = translator(v, src_lang="de", tgt_lang="en")[0]['translation_text']
                except Exception as e:
                    print(f"HuggingFace failed for key {k}: {e}")
                    # Fallback to GoogleTranslator
                    try:
                        translated = GoogleTranslator(source='de', target='en').translate(v)
                    except Exception as e2:
                        print(f"GoogleTranslate also failed for key {k}: {e2}")
                        translated = ""
                new_data[k] = translated

            results_path = file_path.replace('splits_de.json', 'splits.json')
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)
            print(f"Translated and saved English to {results_path}")
