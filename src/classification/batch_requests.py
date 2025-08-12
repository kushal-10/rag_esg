import os
import json
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from src.classification.prompts import get_classifications, create_batch_object
from src.utils.file_utils import load_json

client = OpenAI()
BASE_DIR = os.path.join("data", "texts")
T = 0.4
BATCH_DIR = os.path.join("data", "batches")
if not os.path.exists(BATCH_DIR):
    os.makedirs(BATCH_DIR)


def save_batch(batches: List, batch_num: int):
    jsonl_path = os.path.join(BATCH_DIR, f"batch_{batch_num}.jsonl")

    with open(jsonl_path, 'w') as file:
        for obj in batches:
            file.write(json.dumps(obj) + '\n')


def create_batches(split_paths: List):

    batches = []
    batch_num = 0

    for sp in tqdm(split_paths):
        csv_path = sp.replace("splits.json", "similarity_scores.csv").replace("texts", "scores_csv")
        assert os.path.exists(csv_path), f"{csv_path} does not exist"

        json_data = load_json(sp)
        csv_df = pd.read_csv(csv_path)

        columns = list(csv_df.columns)

        for i in range(len(csv_df)):
            for j in range(1, len(columns)):
                score = csv_df.iloc[i][columns[j]]
                if score >= T:
                    sentence_id = str(int(csv_df.iloc[i][columns[0]]))
                    sentence = json_data[sentence_id]
                    batch_obj = create_batch_object(sentence, sentence_id, csv_path, model="gpt-5-nano")
                    batches.append(batch_obj)

                    if len(batches) >= 40000:
                        save_batch(batches, batch_num)
                        batch_num += 1
                        batches = []
                    break

    save_batch(batches, batch_num)

    print(f"Created {batch_num} batches")




if __name__ == "__main__":

    # 1 - Create Batches
    # splits = []
    # for dirname, _, filenames in os.walk(BASE_DIR):
    #     for filename in filenames:
    #         filepath = os.path.join(dirname, filename)
    #         if filename.endswith("splits.json"):
    #             splits.append(filepath)

    # create_batches(splits)

    # 2 - Submit requests

    batches = os.listdir(BATCH_DIR)
    batches = [os.path.join(BATCH_DIR, batches) for batches in batches if batches.endswith(".jsonl")]

    for i in tqdm(range(len(batches)-1)):

        batch_file = client.files.create(
            file=open(batches[i], "rb"),
            purpose="batch"
        )
        print("/" * 50)
        print(f"BATCH - {batch_file}")
        print("/"*50)

        print(f"batch_file: {batch_file}")
        print(f"batch_file.id: {batch_file.id}")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"batch_job.id: {batch_job.id}")
        print(f"batch_job.output_file_id: {batch_job.output_file_id}")








