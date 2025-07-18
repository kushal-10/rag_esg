import json
import os
from typing import List

import pandas as pd
from tqdm import tqdm

base_dir = os.path.join("data", "textsv2")

with open(os.path.join("data", "sdgs.json"), "r", encoding="utf-8") as f:
    sdgs = json.load(f)

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_ai_threshold_scores(json_data, threshold):
    total_docs = 0
    for kw in json_data:
        kw_dict = json_data[kw]["retrieved_docs"]
        for chunk_data in kw_dict:
            if chunk_data[0] >= threshold:
                total_docs += 1
    return total_docs

def get_sdg_threshold_scores(json_data, threshold, kw):
    total_docs = 0
    kw_dict = json_data[kw]["retrieved_docs"]
    for chunk_data in kw_dict:
        if chunk_data[0] >= threshold:
            total_docs += 1
    return total_docs

def save_csv(rows_data, cols, save_path):
    df = pd.DataFrame(rows_data, columns=cols)
    df.to_csv(save_path, index=False)


def calculate_scores(files: List, threshold: float = 0.6):
    rows_data = []
    cols = ["Company", "Year", "AI_Score", "SDG1_Score", "SDG2_Score", "SDG3_Score", "SDG4_Score",
            "SDG5_Score", "SDG6_Score", "SDG7_Score", "SDG8_Score", "SDG9_Score", "SDG10_Score", "SDG11_Score",
            "SDG12_Score", "SDG13_Score", "SDG14_Score", "SDG15_Score", "SDG16_Score", "SDG17_Score"]

    for file in tqdm(files):
        rag_file_ai = file.replace("embedding_filter.json", "rag_filter_ai.json")
        rag_file_sdg = file.replace("embedding_filter.json", "rag_filter_sdg.json")

        if not os.path.exists(rag_file_ai) or not os.path.exists(rag_file_sdg):
            continue

        rag_data_ai = load_json(rag_file_ai)
        rag_data_sdg = load_json(rag_file_sdg)
        if not rag_data_sdg or not rag_data_ai:
            print("LOL")
            break
        file_splits = file.split("/")
        company_name = file_splits[-3]
        year = file_splits[-2]
        ai_score = get_ai_threshold_scores(rag_data_ai, threshold)

        row_data = [company_name, year, ai_score]

        for sdg_no, sdg_str in sdgs.items():
            kw_score = get_sdg_threshold_scores(rag_data_sdg, threshold, sdg_str)
            row_data.append(kw_score)
            # print(row_data)
        rows_data.append(row_data)

    save_csv(rows_data, cols, os.path.join("resultsv2", f"rag_scores_individual_{str(threshold)}.csv"))

if __name__=="__main__":

    files = []
    for dirname, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith("embedding_filter.json"):
                files.append(os.path.join(dirname, filename))

    calculate_scores(files, 0.5)
    calculate_scores(files, 0.6)
    calculate_scores(files, 0.7)
    calculate_scores(files, 0.75)




