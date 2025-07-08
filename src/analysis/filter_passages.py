import duckdb
import glob
import json
import os
from tqdm import tqdm

# Step 1: Discover all CSVs
csv_files = glob.glob("data/texts/*/*/sentence_scores.csv")

# Step 2: Optionally, build a dict of all splits.json for chunk_id-to-sentence mapping
# You can skip this if you just want chunk_ids. Otherwise, do:
chunkid_to_sentence = {}
for split_path in tqdm(glob.glob("data/texts/*/*/splits.json")):
    with open(split_path, 'r') as f:
        chunkid_to_sentence.update(json.load(f))

# Step 3: Define threshold and target columns
THRESHOLD = 0.4

con = duckdb.connect()

output_records = []

for csv_path in tqdm(csv_files):
    # Create view for easy column reference
    con.execute(f"CREATE VIEW temp_csv AS SELECT * FROM read_csv_auto('{csv_path}')")
    columns = con.execute("PRAGMA table_info('temp_csv')").fetchdf()['name'].tolist()
    # Find AI fuzzy and SDG columns (edit this logic as needed for your real column names)
    ai_fuzzy_cols = [col for col in columns if "Fuzzy" in col or col.lower().startswith("ai_fuzzy")]
    # SDG columns: if you know their names, use that; else, select by index
    sdg_cols = columns[1:18]   # assuming first column is 'chunk_id'
    # Compose query: select rows where any relevant score > threshold
    score_criteria = " OR ".join([f'"{col}" > {THRESHOLD}' for col in ai_fuzzy_cols + sdg_cols])
    select_cols = ", ".join(['chunk_id'] + ai_fuzzy_cols + sdg_cols)
    query = f"""
        SELECT {select_cols}
        FROM read_csv_auto('{csv_path}')
        WHERE {score_criteria}
    """
    df = con.execute(query).fetchdf()
    for _, row in df.iterrows():
        chunk_id = str(row['chunk_id'])
        sentence = chunkid_to_sentence.get(chunk_id, "")   # empty if not found
        output_records.append({
            "chunk_id": chunk_id,
            "sentence": sentence
        })

print(f"Writing {len(output_records)} records to embedding_filter.json...")
with open('embedding_filter.json', 'w', encoding='utf-8') as f:
    json.dump(output_records, f, indent=2, ensure_ascii=False)
print("Done!")
