import os
import glob
import json

import pandas as pd
from tqdm import tqdm

BATCH_OBJS_DIR = "data/batches_41_mini/patched_max_tokens_50"  # original requests
RESULTS_DIR    = "data/batch_results"                          # completed results
OUT_JSON       = os.path.join("src", "classification", "results", "merged_classifications.json")

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines
                continue

# 1) Build a map: custom_id -> user content from original batch objects
custom_to_user = {}

req_files = sorted(glob.glob(os.path.join(BATCH_OBJS_DIR, "batch_*.jsonl")))
for req_path in req_files:
    for obj in iter_jsonl(req_path):
        cid = obj.get("custom_id")
        body = obj.get("body", {})
        msgs = body.get("messages", [])
        # take the last user message content
        user_content = None
        for m in reversed(msgs):
            if m.get("role") == "user":
                user_content = m.get("content")
                break
        if cid:
            custom_to_user[cid] = user_content

# 2) Walk result jsonls and collect rows
rows = []
res_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jsonl")))
for res_path in res_files:
    for obj in tqdm(iter_jsonl(res_path), desc=res_path):
        cid = obj.get("custom_id")
        resp = obj.get("response") or {}
        status_code = resp.get("status_code")
        if status_code != 200:
            # skip non-OK lines (or collect if you want to audit)
            continue

        body = (resp.get("body") or {})
        choices = body.get("choices") or []
        msg = choices[0].get("message") if choices else {}
        assistant_content = (msg or {}).get("content")

        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        rows.append({
            "custom_id": cid,
            "user_content": custom_to_user.get(cid),
            "assistant_content": assistant_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

# 3) Save JSON
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(rows)} rows to {OUT_JSON}")
