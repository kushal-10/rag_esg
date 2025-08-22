# handle data merge
#!/usr/bin/env python3
import os
import re
import json
import glob
import csv

import pandas as pd

# ---- Config ----
BATCH_INPUT_DIR = os.path.join("data", "batches_41_mini", "patched_max_tokens_50")
BATCH_INPUT_PATTERN = os.path.join(BATCH_INPUT_DIR, "batch_[0-4].jsonl")  # skip batch_5
RESULTS_DIR = os.path.join("data", "batch_results")
RESULTS_PATTERN = os.path.join(RESULTS_DIR, "*.jsonl")
OUT_DIR = os.path.join("data", "compiled")
OUT_CSV = os.path.join(OUT_DIR, "batch_merged.csv")

# Regex: task-<numbers>-<company>-<year>
CID_RE = re.compile(r"^task-(?P<number>\d+)-(?P<company>.+)-(?P<year>\d{4})$")

def iter_jsonl(path_pattern):
    for path in sorted(glob.glob(path_pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line=line.strip()
                if not line:
                    continue
                try:
                    yield path, line_no, json.loads(line)
                except json.JSONDecodeError:
                    # tolerate partial/truncated lines
                    continue

def parse_custom_id(custom_id: str):
    m = CID_RE.match(custom_id.strip())
    if not m:
        return None, None, None
    return m.group("number"), m.group("company"), m.group("year")

def collect_inputs():
    rows = []
    for path, line_no, obj in iter_jsonl(BATCH_INPUT_PATTERN):
        custom_id = obj.get("custom_id")
        if not custom_id:
            continue
        number, company, year = parse_custom_id(custom_id)
        if not (number and company and year):
            continue

        # Extract the user sentence from messages
        sentence = None
        try:
            messages = obj.get("body", {}).get("messages", [])
            # pick first user message content
            for m in messages:
                if m.get("role") == "user":
                    sentence = m.get("content")
                    break
        except Exception:
            sentence = None

        rows.append({
            "CustomID": custom_id,
            "SomeNumber": number,
            "Company": company,
            "Year": year,
            "Sentence": sentence
        })
    return pd.DataFrame(rows)

def collect_results():
    recs = {}
    for path, line_no, obj in iter_jsonl(RESULTS_PATTERN):
        custom_id = obj.get("custom_id")
        if not custom_id:
            continue
        resp = obj.get("response") or {}
        status = resp.get("status_code")
        if status != 200:
            continue
        body = resp.get("body") or {}
        # assistant content
        content = None
        try:
            choices = body.get("choices") or []
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content")
        except Exception:
            content = None

        # tokens
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        recs[custom_id] = {
            "AssistantContent": content,
            "PromptTokens": prompt_tokens,
            "CompletionTokens": completion_tokens
        }
    # make a DataFrame
    if not recs:
        return pd.DataFrame(columns=["CustomID","AssistantContent","PromptTokens","CompletionTokens"])
    df = pd.DataFrame.from_dict(recs, orient="index").reset_index().rename(columns={"index":"CustomID"})
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_in = collect_inputs()
    df_res = collect_results()

    # Merge on CustomID
    df = df_in.merge(df_res, on="CustomID", how="left")

    # Reorder/final columns
    cols = [
        "Company", "Year", "SomeNumber", "Sentence",
        "AssistantContent", "PromptTokens", "CompletionTokens",
        "CustomID"
    ]
    df = df[cols]

    # Save
    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,  # safe option
        escapechar="\\",
        lineterminator="\n",  # ✅ correct arg name
    )

    print(f"Saved {len(df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
