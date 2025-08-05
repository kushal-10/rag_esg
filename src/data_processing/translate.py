import json
import os
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

# NOTE: batch_size=16 is a good start; tune this for your GPU
translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if device != "cpu" else -1)

root_dir = 'data/textsv3'
BATCH_SIZE = 16  # Tune as needed, depending on sentence length and GPU RAM

def chunk_text(text, max_tokens=500):
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

def batch_translate(texts, src_lang="de", tgt_lang="en", max_length=500):
    try:
        # Call pipeline with a list (batch)
        outputs = translator(
            texts, src_lang=src_lang, tgt_lang=tgt_lang, max_length=max_length
        )
        # Outputs is a list of dicts with 'translation_text'
        return [o['translation_text'] for o in outputs]
    except Exception as e:
        print(f"Batch translation failed: {e}")
        # Fallback: try Google Translate in serial
        fallback = []
        for v in texts:
            try:
                fallback.append(GoogleTranslator(source=src_lang, target=tgt_lang).translate(v))
            except Exception as e2:
                print(f"Google Translate also failed: {e2}")
                fallback.append("")
        return fallback

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith('splits_de.json'):
            file_path = os.path.join(root, file)
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                json_data = json.load(f)
            new_data = {}

            keys = list(json_data.keys())
            values = list(json_data.values())

            batch_texts = []
            batch_keys = []
            result_map = {}

            for k, v in tqdm(zip(keys, values), total=len(keys)):
                try:
                    tokens = tokenizer(v, return_tensors="pt")["input_ids"]
                    if tokens.shape[1] > 500:
                        # Need to chunk and process chunks separately
                        chunks = chunk_text(v, max_tokens=500)
                        translated_chunks = batch_translate(chunks)
                        translated = " ".join(translated_chunks)
                        new_data[k] = translated
                    else:
                        batch_texts.append(v)
                        batch_keys.append(k)
                        # When enough for a batch, translate them
                        if len(batch_texts) == BATCH_SIZE:
                            translations = batch_translate(batch_texts)
                            for bk, bt in zip(batch_keys, translations):
                                new_data[bk] = bt
                            batch_texts = []
                            batch_keys = []
                except Exception as e:
                    print(f"Failed for key {k}: {e}")
                    # Try fallback for this key
                    try:
                        translated = GoogleTranslator(source='de', target='en').translate(v)
                    except Exception as e2:
                        print(f"GoogleTranslate also failed for key {k}: {e2}")
                        translated = ""
                    new_data[k] = translated

            # Translate any remaining texts in the last batch
            if batch_texts:
                translations = batch_translate(batch_texts)
                for bk, bt in zip(batch_keys, translations):
                    new_data[bk] = bt

            results_path = file_path.replace('splits_de.json', 'splits.json')
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)
            print(f"Translated and saved English to {results_path}")
