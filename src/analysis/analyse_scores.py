import os
import pandas as pd

IN_CSV  = os.path.join("src", "classification", "results", "company_year_pos_minus_neg.csv")
OUT_CSV = os.path.join("src", "classification", "results", "entries_all_zero_1_17.csv")

# Load
df = pd.read_csv(IN_CSV)

# Ensure numeric for label cols 1..17 (coerce and fill NaNs as 0)
label_cols = [str(i) for i in range(1, 18)]
for c in label_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Find rows where ALL 1..17 are exactly zero
mask_all_zero = (df[label_cols] == 0).all(axis=1)
offenders = df[mask_all_zero].copy()

# Print summary and a quick peek
print(f"Total rows: {len(df)}")
print(f"Rows with all labels 1..17 == 0: {mask_all_zero.sum()}")

if not offenders.empty:
    print("\nSample (up to 20):")
    print(offenders[["company", "year"] + label_cols].head(20).to_string(index=False))

# Save offending rows
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
offenders.to_csv(OUT_CSV, index=False)
print(f"\nSaved offending rows to: {OUT_CSV}")
