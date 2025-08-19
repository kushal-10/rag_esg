import os
import pandas as pd

# ---- Paths ----
IN_CSV  = os.path.join("src", "classification", "results", "company_year_sentiment_counts.csv")
OUT_CSV = os.path.join("src", "classification", "results", "company_year_pos_minus_neg.csv")

# ---- Config ----
LABEL_COLS = [str(i) for i in range(0, 18)]  # "0".."17"
EXTRA_COLS = ["AI"]                           # keep AI and also subtract POS-NEG
VALUE_COLS = LABEL_COLS + EXTRA_COLS          # all numeric cols to diff

# ---- Load ----
df = pd.read_csv(IN_CSV)

# Normalize sentiment just in case
df["sentiment"] = df["sentiment"].str.strip().str.capitalize()

# If there are accidental dups per (company, year, sentiment), sum them first
df = (
    df.groupby(["company", "year", "sentiment"], as_index=False)[VALUE_COLS]
      .sum()
)

# Split POS / NEG
pos = df[df["sentiment"] == "Positive"].set_index(["company", "year"])
neg = df[df["sentiment"] == "Negative"].set_index(["company", "year"])

# Ensure both sides have the same index and columns; fill missing with 0
all_index = pos.index.union(neg.index)
pos = pos.reindex(all_index, fill_value=0)
neg = neg.reindex(all_index, fill_value=0)

# Safety: keep only numeric columns we expect
pos = pos.reindex(columns=VALUE_COLS, fill_value=0)
neg = neg.reindex(columns=VALUE_COLS, fill_value=0)

# POS - NEG
diff = pos.subtract(neg)

# Drop column "0" as requested
if "0" in diff.columns:
    diff = diff.drop(columns=["0"])

# Bring back columns and write
out = diff.reset_index()

# Optional ordering: company, year, then labels 1..17, then AI
ordered_cols = ["company", "year"] + [str(i) for i in range(1, 18)] + ["AI"]
out = out.reindex(columns=ordered_cols)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print(f"Wrote POS-NEG merged CSV to: {OUT_CSV}")
print(f"Rows (company-year): {len(out)}")
