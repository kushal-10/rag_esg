import duckdb
import glob

# Find one CSV to extract columns
sample_file = glob.glob("data/texts/*/*/sentence_scores.csv")[0]
con = duckdb.connect()

con.execute(f"CREATE VIEW temp_csv AS SELECT * FROM read_csv_auto('{sample_file}')")
pragma_info = con.execute("PRAGMA table_info('temp_csv')").fetchdf()
all_columns = list(pragma_info['name'])

# Identify fuzzy columns for AI (e.g., those ending with "Fuzzy")
ai_fuzzy_cols = [col for col in all_columns if col.endswith("Fuzzy")]
# SDG columns: skip 'chunk_id', take next 17 columns
sdg_columns = all_columns[1:18]  # 1:18 is columns 1-17 (Python slices end-exclusive)

threshold = 0.4  # used for SDG embedding columns

# Build AI and SDG score expressions
ai_score_expr = " + ".join([f'"{col}"' for col in ai_fuzzy_cols])
sdg_score_expr = " + ".join([
    f'CASE WHEN "{col}" > {threshold} THEN 1 ELSE 0 END'
    for col in sdg_columns
])

input_pattern = "data/texts/*/*/sentence_scores.csv"
output_file = f"results/embedding_scores_{threshold}.csv"
query = f"""
WITH all_chunks AS (
  SELECT 
    *,
    REGEXP_EXTRACT(filename, 'data/texts/([^/]+)/([^/]+)/sentence_scores.csv', 1) AS Company,
    REGEXP_EXTRACT(filename, 'data/texts/([^/]+)/([^/]+)/sentence_scores.csv', 2) AS Year,
    ({ai_score_expr}) AS ai_score_chunk,
    ({sdg_score_expr}) AS sdg_score_chunk
  FROM read_csv_auto('{input_pattern}', filename=true)
)
SELECT
  Company,
  Year,
  SUM(ai_score_chunk) AS AI_Score,
  SUM(sdg_score_chunk) AS SDG_Score
FROM all_chunks
GROUP BY Company, Year
ORDER BY Company, Year
"""

con.execute(f"COPY ({query}) TO '{output_file}' (HEADER, DELIMITER ',');")
print(f"Saved results to {output_file}")
