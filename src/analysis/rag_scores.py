import json
import os

import pandas as pd

base_dir = os.path.join("data", "texts")

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_threshold_scores(json_data, threshold):
    total_docs = 0
    for kw in json_data:
        kw_dict = json_data[kw]["retrieved_docs"]
        for chunk_data in kw_dict:
            if chunk_data[0] >= threshold:
                total_docs += 1
    return total_docs

def save_csv(ai_scores, sdg_scores, companies, years, save_path):
    data = {
        "Company": companies,
        "Year": years,
        "AI_Score": ai_scores,
        "SDG_Score": sdg_scores
    }

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


files = []
for dirname, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith("embedding_filter.json"):
            files.append(os.path.join(dirname, filename))

companies = []
years = []
ai_scores = []
sdg_scores = []

ai_scores_06 = []
sdg_scores_06 = []

ai_scores_07 = []
sdg_scores_07 = []

ai_scores_075 = []
sdg_scores_075 = []

for file in files:
    rag_file_ai = file.replace("embedding_filter.json", "rag_filter_ai.json")
    rag_file_sdg = file.replace("embedding_filter.json", "rag_filter_sdg.json")

    if not os.path.exists(rag_file_ai) or not os.path.exists(rag_file_sdg):
        continue

    rag_data_ai = load_json(rag_file_ai)
    rag_data_sdg = load_json(rag_file_sdg)
    print(file)
    if not rag_data_sdg or not rag_data_ai:
        print("LOL")
        break
    file_splits = file.split("/")
    company_name = file_splits[-3]
    year = file_splits[-2]

    companies.append(company_name)
    years.append(year)

    ai_score = 0
    for kw in rag_data_ai.keys():
        kw_dict = rag_data_ai[kw]
        ai_score += kw_dict["total_docs"]

    sdg_score = 0
    for kw in rag_data_sdg.keys():
        kw_dict = rag_data_sdg[kw]
        sdg_score += kw_dict["total_docs"]

    ai_scores.append(ai_score)
    sdg_scores.append(sdg_score)

    ai_scores_06.append(get_threshold_scores(rag_data_ai, 0.6))
    sdg_scores_06.append(get_threshold_scores(rag_data_sdg, 0.6))
    ai_scores_07.append(get_threshold_scores(rag_data_ai, 0.7))
    sdg_scores_07.append(get_threshold_scores(rag_data_sdg, 0.7))
    ai_scores_075.append(get_threshold_scores(rag_data_ai, 0.75))
    sdg_scores_075.append(get_threshold_scores(rag_data_sdg, 0.75))


save_csv(ai_scores, sdg_scores, companies, years, os.path.join("results", "rag_scores_0.5.csv"))
save_csv(ai_scores_06, sdg_scores_06, companies, years, os.path.join("results", "rag_scores_0.6.csv"))
save_csv(ai_scores_07, sdg_scores_07, companies, years, os.path.join("results", "rag_scores_0.7.csv"))
save_csv(ai_scores_075, sdg_scores_075, companies, years, os.path.join("results", "rag_scores_0.75.csv"))
