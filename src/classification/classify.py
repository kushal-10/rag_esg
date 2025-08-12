import os
import json

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from src.classification.prompts import BASE

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-5-nano"  # Replace with actual model name
T = 0.4

PROMPT_TEMPLATE = BASE

def classify_sentence(sentence):
    """Send prompt to GPT and return classification."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(sentence=sentence)}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with GPT call: {e}")
        return None

def process_file(csv_path, json_path, save_path):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    df = pd.read_csv(csv_path)

    results = []
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        for j in range(1, len(row_data)):
            val = row_data.iloc[j]
            if val > T:
                row_id = str(int(row_data.iloc[0]))
                sentence_text = json_data[row_id]
                output = classify_sentence(sentence_text)

                results.append({
                    "sentence_id": row_id,
                    "classification": output
                })
                break

    pd.DataFrame(results).to_csv(save_path, index=False)

if __name__ == "__main__":
    base_dir = os.path.join("data", "scores_csv")
    output_dir = "data/sdg_sentiment_results"
    os.makedirs(output_dir, exist_ok=True)

    csvs = []
    for dirname, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                csvs.append(os.path.join(dirname, filename))

    for csv_file in tqdm(csvs):
        print(f"Processing {csv_file}")
        json_path = csv_file.replace("similarity_scores.csv", "splits.json")
        json_path = json_path.replace("scores_csv", "texts")

        save_path = csv_file.replace("similarity_scores.csv", "classifications.csv")

        if not os.path.exists(save_path):
            process_file(csv_file, json_path, save_path)