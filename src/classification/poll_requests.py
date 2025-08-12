# poll_results.py
import os
import re
import json
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ----------------- CONFIG -----------------
BASE_DIR_SCORES = os.path.join("data", "scores_csv")
RESULTS_ROOT = "data/sdg_sentiment_results"

BATCH_LOG = "batch_ids.json"
MAPPINGS_DIR = "mappings"
POLL_SECS = 6
# ------------------------------------------

client = OpenAI()

def index_csvs_by_basename(base_dir: str):
    """Map filename basename -> full csv_path(s). Warn if duplicates."""
    idx = defaultdict(list)
    for dirname, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.endswith(".csv"):
                idx[fn].append(os.path.join(dirname, fn))
    return idx

def load_existing_results(csv_path: str):
    out_path = csv_path.replace("similarity_scores.csv", "classifications.csv")
    if os.path.exists(out_path):
        try:
            df = pd.read_csv(out_path, dtype={"sentence_id": str, "classification": str})
            return out_path, df
        except Exception:
            return out_path, pd.DataFrame(columns=["sentence_id", "classification"])
    else:
        # ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return out_path, pd.DataFrame(columns=["sentence_id", "classification"])

def poll_and_collect(batch_ids):
    """Yield (batch_id, status, output_file_id) when terminal."""
    remaining = set(batch_ids)
    while remaining:
        finished = []
        for bid in list(remaining):
            b = client.batches.retrieve(bid)
            if b.status in ("completed", "failed", "cancelled", "expired"):
                finished.append((bid, b.status, getattr(b, "output_file_id", None)))
                remaining.remove(bid)
        for item in finished:
            yield item
        if remaining:
            time.sleep(POLL_SECS)

if __name__ == "__main__":
    if not os.path.exists(BATCH_LOG):
        raise SystemExit(f"Missing {BATCH_LOG}. Run submit_batches.py first.")

    with open(BATCH_LOG, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Only poll batches that don't yet have an "output_downloaded" flag
    pending = [e for e in entries if not e.get("done")]
    if not pending:
        print("No pending batches. All done.")
        raise SystemExit(0)

    # Build index: basename -> full paths (in case same basename appears in multiple dirs)
    basename_index = index_csvs_by_basename(BASE_DIR_SCORES)

    # Poll
    id_list = [e["batch_id"] for e in pending]
    for bid, status, out_fid in tqdm(poll_and_collect(id_list), total=len(id_list), desc="Polling batches"):
        # Update entry in memory
        for e in entries:
            if e["batch_id"] == bid:
                e["status"] = status

        if status != "completed" or not out_fid:
            print(f"[WARN] Batch {bid} ended with status={status}")
            for e in entries:
                if e["batch_id"] == bid:
                    e["done"] = True
            continue

        # Download output
        out = client.files.content(out_fid)
        lines = out.text.strip().splitlines()

        # mapping file for this batch
        mapping_path = os.path.join(MAPPINGS_DIR, f"{bid}.jsonl")
        if not os.path.exists(mapping_path):
            print(f"[WARN] Missing mapping for batch {bid}; attempting fallback from custom_id.")
        # Collect results per csv_path
        per_file_rows = defaultdict(list)

        for line in lines:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            resp = obj.get("response", {})
            body = resp.get("body", {})
            try:
                content = body["choices"][0]["message"]["content"].strip()
            except Exception:
                content = f"[ERROR] {body}"

            # Recover csv target and sentence_id from custom_id format "<basename>:<sent_id>"
            m = re.match(r"^(.*\.csv):(\d+)$", cid)
            if not m:
                continue
            basename, sent_id = m.group(1), m.group(2)

            # Resolve full path (prefer unique match)
            candidates = basename_index.get(basename, [])
            if not candidates:
                # couldn't resolve; skip
                continue
            if len(candidates) > 1:
                # If ambiguous, pick the one whose classifications.csv is missing or smaller
                # (heuristic; usually basenames are unique in your tree)
                target = sorted(candidates, key=lambda p: os.path.getsize(
                    p.replace("similarity_scores.csv","classifications.csv")) if os.path.exists(
                    p.replace("similarity_scores.csv","classifications.csv")) else -1
                )[-1]
            else:
                target = candidates[0]

            per_file_rows[target].append({"sentence_id": sent_id, "classification": content})

        # Merge & write per file
        for csv_path, new_rows in per_file_rows.items():
            out_path, existing = load_existing_results(csv_path)
            df_new = pd.DataFrame(new_rows)
            if not existing.empty:
                merged = pd.concat([existing, df_new], ignore_index=True)
                merged.drop_duplicates(subset=["sentence_id"], keep="last", inplace=True)
            else:
                merged = df_new.drop_duplicates(subset=["sentence_id"], keep="last")
            merged.to_csv(out_path, index=False)

        # mark batch done
        for e in entries:
            if e["batch_id"] == bid:
                e["done"] = True
                e["output_file_id"] = out_fid

        # persist progress so you can kill/restart safely
        with open(BATCH_LOG, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    print("Polling complete.")
