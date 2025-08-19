import os
import re
import json
import ast
import csv
from collections import defaultdict
from tqdm import tqdm

RESULTS_JSON = os.path.join("src", "classification", "results", "merged_classifications.json")
OUT_CSV      = os.path.join("src", "classification", "results", "company_year_sentiment_counts.csv")

LABEL_MIN = 0
LABEL_MAX = 17
LABELS = list(range(LABEL_MIN, LABEL_MAX + 1))  # 0..17 inclusive

# ---------- Robust parsing for assistant_content ----------
def _coerce_token(tok: str):
    t = tok.strip().strip('"').strip("'")
    if t.lower() == "true": return True
    if t.lower() == "false": return False
    try:
        if re.fullmatch(r"[+-]?\d+", t):
            return int(t)
        if (re.fullmatch(r"[+-]?\d*\.\d+(e[+-]?\d+)?", t, flags=re.I) or
            re.fullmatch(r"[+-]?\d+e[+-]?\d+", t, flags=re.I)):
            return float(t)
    except Exception:
        pass
    return t

def parse_assistant_content(s: str):
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    if not (s.startswith("[") and s.endswith("]")):
        m = re.search(r"\[(.*)\]", s, flags=re.S)
        if m:
            s = "[" + m.group(1) + "]"
        else:
            parts = [p for p in re.split(r",(?![^\"']*[\"'])", s)]
            return [_coerce_token(p) for p in parts if p.strip()]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    inner = s[1:-1]
    parts = [p for p in re.split(r",(?![^\"']*[\"'])", inner)]
    return [_coerce_token(p) for p in parts if p.strip()]

# ---------- NEW: company/year extraction per your rule ----------
def extract_company_year(custom_id: str):
    """
    Examples:
      task-16-17.e.on_$38.46 b_energy-2014         -> company='e.on_$38.46 b_energy', year='2014'
      task-16-18.henkel_$35.64 b_consumer staplers-2016
      task-16-19.hannover rück_$33.67 b_financial service-2021
      task-16-22.airbus-2019
      task-16-25.continental-2021
      task-16-32.rwe-2016
      task-16-32.rwe-2017
      task-16-34.vonovia-2015
      task-16-34.vonovia-2017
      task-16-34.vonovia-2020
      task-16-34.vonovia-2021
      task-16-7.porsche_$67.40 b_consumer discretionary-2017
      task-16-8.munich re_$63.39_financials-2018
      task-16-8.munich re_$63.39_financials-2019
      task-1910-28.freseniusmedicalcare-2023
      task-2012-6.mercedes-benz_$68.14 b_consumer discretionary-2018

    Rule:
      - YEAR = last four digits at the end (…-YYYY)
      - Remove leading 'task-' + [digits and hyphens] + ('.' or '-') once
      - The remainder (before -YYYY) is the company name (kept as-is)
    """
    if not custom_id:
        return None, None

    s = custom_id.strip()

    # 1) year at the very end: "-YYYY"
    m_year = re.search(r"-(\d{4})$", s)
    if not m_year:
        return None, None
    year = m_year.group(1)

    # everything before "-YYYY"
    base = s[:m_year.start()]

    # 2) strip one leading "task-<digits/hyphens><dot or hyphen>"
    #    handles "task-16-32.rwe", "task-16-7.porsche…", "task-1910-28.fresenius…"
    base = re.sub(r"^task-[0-9-]+[.\-]", "", base, count=1)

    company = base.strip()
    if not company:
        return None, None

    return company, year

def normalize_sentiment(s):
    if s is None:
        return None
    t = str(s).strip().lower()
    if "pos" in t:
        return "Positive"
    if "neg" in t:
        return "Negative"
    return None

def core_is_malformed(core):
    if not isinstance(core, (list, tuple)):
        return True
    if len(core) == 0:
        return True  # treat empty as malformed; change if you prefer otherwise
    for v in core:
        if not isinstance(v, int):
            return True
        if v < LABEL_MIN or v > LABEL_MAX:
            return True
    return False

# ---------- Aggregation ----------
agg = defaultdict(lambda: {str(k): 0 for k in LABELS} | {"AI": 0})

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    rows = json.load(f)

skipped_no_cy = 0
skipped_no_sent = 0

for row in tqdm(rows, total=len(rows)):
    cid = row.get("custom_id")
    company, year = extract_company_year(cid)
    if not company or not year:
        skipped_no_cy += 1
        continue

    parsed = parse_assistant_content(row.get("assistant_content"))
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 1:
        skipped_no_sent += 1
        continue

    ai_flag = parsed[-2] if len(parsed) >= 2 else None
    sentiment = parsed[-1] if len(parsed) >= 1 else None
    sentiment = normalize_sentiment(sentiment)
    if sentiment is None:
        skipped_no_sent += 1
        continue

    key = (company, year, sentiment)
    bucket = agg[key]

    core = list(parsed[:-2]) if len(parsed) > 2 else []

    # Rule 1: 0 present OR malformed -> increment only column "0"
    if (0 in core) or core_is_malformed(core):
        bucket["0"] += 1
    else:
        # Rule 2: duplicates count once
        for v in set(core):
            if isinstance(v, int) and LABEL_MIN <= v <= LABEL_MAX:
                bucket[str(v)] += 1

    # AI column
    if isinstance(ai_flag, bool) and ai_flag is True:
        bucket["AI"] += 1

# ---------- Write CSV ----------
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
fieldnames = ["company", "year", "sentiment"] + [str(k) for k in LABELS] + ["AI"]

def sentiment_order(s): return 0 if s == "Positive" else 1
sorted_keys = sorted(agg.keys(), key=lambda k: (k[0], k[1], sentiment_order(k[2])))

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for (company, year, sentiment) in sorted_keys:
        row = {"company": company, "year": year, "sentiment": sentiment}
        row.update(agg[(company, year, sentiment)])
        writer.writerow(row)

print(f"Wrote aggregated CSV to: {OUT_CSV}")
print(f"Buckets created: {len(agg)} (company-year-sentiment)")
print(f"Skipped (no company/year): {skipped_no_cy}")
print(f"Skipped (no/unknown sentiment): {skipped_no_sent}")
