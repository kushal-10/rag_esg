import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("src/classification/results/company_year_sentiment_counts.csv")

# Columns of interest: SDG 1–17 and AI
sdg_cols = [str(i) for i in range(1, 18)] + ["AI"]

# Group by sentiment and sum counts
grouped = df.groupby("sentiment")[sdg_cols].sum()

# Ensure Positive and Negative rows exist
for sentiment in ["Positive", "Negative"]:
    if sentiment not in grouped.index:
        grouped.loc[sentiment] = 0

# Extract counts
positive_counts = grouped.loc["Positive"]
negative_counts = -grouped.loc["Negative"]  # flip for mirror effect

# Plot
x = range(len(sdg_cols))
plt.figure(figsize=(12, 6))

plt.bar(x, positive_counts, color="green", label="Positive")
plt.bar(x, negative_counts, color="red", label="Negative")

# Formatting
plt.axhline(0, color="black", linewidth=1)
plt.xticks(x, sdg_cols, rotation=45)
plt.ylabel("Counts")
plt.xlabel("SDGs and AI")
plt.title("Positive vs Negative Counts per SDG and AI")
plt.legend()
plt.tight_layout()
plt.show()
