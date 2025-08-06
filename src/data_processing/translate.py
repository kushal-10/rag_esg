import json
import os
import torch
import threading
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "facebook/m2m100_418M"
SRC_LANG = "de"
TGT_LANG = "en"
ROOT_DIR = os.path.abspath("textsv3")
BATCH_SIZE = 128  # Initial batch size (auto-scales if OOM)
MAX_TOKENS = 400  # Lowered for safe padding under 1024
USE_FP16 = True
TIMEOUT = 30  # seconds per batch
LONG_TEXT_WORDS = 400  # Chunk very long texts by words

# ==============================
# DEVICE SETUP
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ==============================
# LOAD MODEL & TOKENIZER
# ==============================
print(f"📥 Loading model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if device == "cuda":
    if USE_FP16:
        model.half()
    model.to(device)
    try:
        model = torch.compile(model)  # Requires PyTorch 2.6+
    except Exception as e:
        print(f"⚠ torch.compile() not available: {e}")

model.eval()

# ==============================
# HELPER FUNCTIONS
# ==============================
def chunk_by_words(text, chunk_size=LONG_TEXT_WORDS):
    """Split text by words into chunks of size chunk_size."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def chunk_text_by_tokens(text, max_tokens=MAX_TOKENS):
    """Split text into chunks respecting token count limit."""
    words = text.split()
    chunks, current_chunk = [], []
    for word in words:
        current_chunk.append(word)
        tokens = tokenizer(" ".join(current_chunk), return_tensors="pt")["input_ids"]
        if tokens.shape[1] >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def batch_translate(texts, src_lang=SRC_LANG, tgt_lang=TGT_LANG):
    """Translate a batch of texts using optimized settings."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS
    ).to(device)

    dynamic_max_length = min(MAX_TOKENS, inputs["input_ids"].shape[1] + 30)

    with torch.inference_mode():
        if USE_FP16 and device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                    max_length=dynamic_max_length,
                    num_beams=1  # greedy decoding for speed
                )
        else:
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                max_length=dynamic_max_length,
                num_beams=1
            )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def safe_batch_translate(texts):
    """Retry with smaller batch size on OOM."""
    size = len(texts)
    while size > 0:
        try:
            return batch_translate(texts[:size])
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                size = size // 2
                print(f"⚠ OOM detected. Reducing batch size to {size}")
            else:
                raise e
    return []

def run_with_timeout(func, args=(), timeout=TIMEOUT):
    """Run function with timeout, return [] if it hangs."""
    result = []
    def target():
        nonlocal result
        result = func(*args)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print("⚠ Timeout reached! Skipping this batch.")
        return []
    return result

# ==============================
# MAIN LOOP
# ==============================
if not os.path.exists(ROOT_DIR):
    print(f"❌ Directory does not exist: {ROOT_DIR}")
    exit()
else:
    print(f"✅ Found directory: {ROOT_DIR}")

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith("splits_de.json"):
            file_path = os.path.join(root, file)
            print(f"\n📂 Processing: {file_path}")

            with open(file_path, encoding="utf-8", errors="ignore") as f:
                json_data = json.load(f)

            new_data = {}
            keys = list(json_data.keys())
            values = list(json_data.values())

            batch_texts, batch_keys = [], []

            for k, v in tqdm(zip(keys, values), total=len(keys), desc="Translating"):
                try:
                    print(f"🔍 Key: {k}, Length: {len(v.split())} words")

                    # If text is very long, chunk early by words
                    if len(v.split()) > LONG_TEXT_WORDS * 2:  # >800 words
                        print(f"⚠ Long text detected for key {k}, chunking by words...")
                        chunks = chunk_by_words(v, LONG_TEXT_WORDS)
                        translated_chunks = []
                        for c in chunks:
                            translated_chunks.extend(run_with_timeout(safe_batch_translate, ([c],)))
                        new_data[k] = " ".join(translated_chunks)
                        continue

                    # Check token limit
                    tokens = tokenizer(v, return_tensors="pt", truncation=False)["input_ids"]
                    if tokens.shape[1] > 1024:
                        print(f"⚠ Key {k} exceeds model limit ({tokens.shape[1]} tokens), chunking by tokens...")
                        chunks = chunk_text_by_tokens(v, max_tokens=MAX_TOKENS)
                        translated_chunks = []
                        for c in chunks:
                            translated_chunks.extend(run_with_timeout(safe_batch_translate, ([c],)))
                        new_data[k] = " ".join(translated_chunks)
                        continue

                    # Normal case: batch it
                    batch_texts.append(v)
                    batch_keys.append(k)

                    if len(batch_texts) >= BATCH_SIZE:
                        translations = run_with_timeout(safe_batch_translate, (batch_texts,))
                        for bk, bt in zip(batch_keys, translations):
                            new_data[bk] = bt
                        batch_texts, batch_keys = [], []

                except Exception as e:
                    print(f"❌ Failed for key {k}: {e}")
                    try:
                        translated = GoogleTranslator(source=SRC_LANG, target=TGT_LANG).translate(v)
                    except Exception as e2:
                        print(f"❌ GoogleTranslate also failed for {k}: {e2}")
                        translated = ""
                    new_data[k] = translated

            # Flush remaining
            if batch_texts:
                translations = run_with_timeout(safe_batch_translate, (batch_texts,))
                for bk, bt in zip(batch_keys, translations):
                    new_data[bk] = bt

            results_path = file_path.replace("splits_de.json", "splits.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)
            print(f"✅ Translated and saved to: {results_path}")
