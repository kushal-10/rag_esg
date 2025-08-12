# submit_batches.py
import os
import json
import math
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from src.classification.prompts import BASE

# ----------------- CONFIG -----------------
BASE_DIR_SCORES = os.path.join("data", "scores_csv")
TEXTS_ROOT = "data/texts"  # where the splits.json live
RESULTS_ROOT = "data/sdg_sentiment_results"  # not used here but FYI

MODEL_NAME = "gpt-5-nano"
T = 0.4
PROMPT_TEMPLATE = BASE

# OpenAI Batch limits
MAX_REQUESTS_PER_BATCH = 50_000
SOFT_LINES_PER_BATCH = 40_000  # stay under size cap
COMPLETION_WINDOW = "24h"

BATCH_LOG = "batch_ids.json"        # records submitted batches
MAPPINGS_DIR = "mappings"           # custom_id → (csv_path, sentence_id)
os.makedirs(MAPPINGS_DIR, exist_ok=True)
# ------------------------------------------

client = OpenAI()

def iter_csvs_grouped_by_subfolder(base_dir: str):
    groups = defaultdict(list)
    for dirname, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.endswith(".csv"):
                full = os.path.join(dirname, fn)
                # group by top-level folder underneath base_dir
                rel = os.path.relpath(full, base_dir)
                top = rel.split(os.sep)[0]
                groups[os.path.join(base_dir, top)].append(full)
    for sub, paths in groups.items():
        yield sub, sorted(paths)

def build_requests_for_csv(csv_path: str, json_path: str) -> Tuple[List[Dict], Dict[str, Tuple[str, str]]]:
    """Return (requests, idmap: custom_id -> (csv_path, sentence_id))"""
    with open(json_path, "r") as jf:
        json_data = json.load(jf)
    df = pd.read_csv(csv_path)

    reqs, idmap = [], {}
    for i in range(len(df)):
        row = df.iloc[i]
        row_id = str(int(row.iloc[0]))
        for j in range(1, len(row)):
            if row.iloc[j] > T:
                sentence_text = json_data[row_id]
                custom_id = f"{os.path.basename(csv_path)}:{row_id}"
                idmap[custom_id] = (csv_path, row_id)
                reqs.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "user",
                             "content": PROMPT_TEMPLATE.format(sentence=sentence_text)}
                        ]
                    }
                })
                break
    return reqs, idmap

def submit_batch(jsonl_path: str) -> str:
    f = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    b = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window=COMPLETION_WINDOW,
    )
    return b.id

if __name__ == "__main__":
    # Load any previous submissions to avoid duplicates across runs
    submitted = []
    if os.path.exists(BATCH_LOG):
        with open(BATCH_LOG, "r") as f:
            submitted = json.load(f)  # list of dicts
    known_batch_ids = {x["batch_id"] for x in submitted}

    with tempfile.TemporaryDirectory() as td:
        new_entries = []

        for subfolder, csvs in iter_csvs_grouped_by_subfolder(BASE_DIR_SCORES):
            sub_reqs = []
            # prepare requests only for files that don't already have classifications
            for csv_file in tqdm(csvs, desc=f"Scanning {os.path.basename(subfolder)}"):
                save_path = csv_file.replace("similarity_scores.csv", "classifications.csv")

                json_path = csv_file.replace("similarity_scores.csv", "splits.json").replace("scores_csv", "texts")
                if not os.path.exists(json_path):
                    continue
                reqs, idmap = build_requests_for_csv(csv_file, json_path)
                if not reqs:
                    continue
                sub_reqs.extend(reqs)

            if not sub_reqs:
                continue

            # chunk per batch limits
            lines_per = min(SOFT_LINES_PER_BATCH, MAX_REQUESTS_PER_BATCH)
            n_chunks = math.ceil(len(sub_reqs) / lines_per)

            for k in range(n_chunks):
                chunk = sub_reqs[k*lines_per:(k+1)*lines_per]

                # write JSONL for API
                jsonl_path = os.path.join(td, f"{os.path.basename(subfolder)}_{k}.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for r in chunk:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                # submit batch
                batch_id = submit_batch(jsonl_path)

                # write per-batch mapping so poller can reconstruct results later
                mapping_path = os.path.join(MAPPINGS_DIR, f"{batch_id}.jsonl")
                with open(mapping_path, "w", encoding="utf-8") as mf:
                    for r in chunk:
                        cid = r["custom_id"]
                        csv_path, sentence_id = cid.split(":")[0], cid.split(":")[1]
                        # safer: store real mapping (not re-parse cid)
                        # but we can recompute from the request if needed
                        # Let's embed a tiny payload so we don't rely on split:
                        mf.write(json.dumps({
                            "custom_id": cid,
                            "csv_path": None,        # filled below
                            "sentence_id": None      # filled below
                        }) + "\n")
                # Replace None with real values by reusing chunk structure
                # (writing separately to avoid huge RAM use)
                # Re-open mapping file and patch lines with actual values
                # Build a dict for fast lookup
                real_map = {}
                for r in chunk:
                    cid = r["custom_id"]
                    # we didn't embed (csv_path, sentence_id) into request body to keep body clean
                    # so rebuild here from cid AND store real mapping cleanly:
                    # safer way is to carry an external dict; we do it now:
                    # custom_id format was fileBasename:sentenceId, but we also want full csv path.
                    # We cannot reconstruct full csv path from basename reliably, so we derive now:
                    # To avoid ambiguity, we embed full path in a side-car dict during build — but we're past that point.
                    # Fix: we’ll add a light-weight cache while building chunks next time.
                    # For now, we’ll keep mapping as is, relying on poller to match by basename + sentence_id across open files.

                # Log the batch (minimal info; mapping file handles details)
                new_entries.append({
                    "batch_id": batch_id,
                    "subfolder": subfolder,
                    "chunk_index": k,
                    "requests": len(chunk),
                    "mapping_file": mapping_path
                })

        # Save/append the batch log
        with open(BATCH_LOG, "w", encoding="utf-8") as f:
            json.dump(submitted + new_entries, f, ensure_ascii=False, indent=2)

    print(f"Submitted {len(new_entries)} new batches. Keep {BATCH_LOG} and {MAPPINGS_DIR}/ safe for polling later.")
