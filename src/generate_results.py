import json
import os
import pandas as pd

pth  = "results/retrieve/14.BASF_$42.93 B_Industrials/results_ai.json"

def get_scores(results_path: str):
    with open(results_path, 'r') as f:
        data = json.load(f)

    scores = {}
    year_keys = list(data.keys())
    for year_key in year_keys:
        total_score = 0
        year_data = data[year_key]
        i = 0
        sdg_scores = {}
        for sdg_dict in year_data:
            i += 1
            sdg_score_count = 0
            sdg_keys = list(sdg_dict.keys())
            for sdg_key in sdg_keys:
                if sdg_key.endswith("docs"):
                    sub_sdg_item = sdg_dict[sdg_key]
                    total_score += len(sub_sdg_item)
                    sdg_score_count += len(sub_sdg_item)

            sdg_scores[str(i)] = sdg_score_count

        scores[year_key] = [total_score, sdg_scores]

    return scores

def get_ai_scores(results_path: str):
    with open(results_path, 'r') as f:
        data = json.load(f)
    scores = {}
    # print(list(data.keys()))
    for k in list(data.keys()):
        sub_keys = list(data[k].keys())
        scores[k] = int(sub_keys[0])

    return scores


def get_sdg_scores():
    df = {
        "company": [],
        "2014_ai": [],
        "2015_ai": [],
        "2016_ai": [],
        "2017_ai": [],
        "2018_ai": [],
        "2019_ai": [],
        "2020_ai": [],
        "2021_ai": [],
        "2022_ai": [],
        "2023_ai": [],
        "2014_sdg": [],
        "2015_sdg": [],
        "2016_sdg": [],
        "2017_sdg": [],
        "2018_sdg": [],
        "2019_sdg": [],
        "2020_sdg": [],
        "2021_sdg": [],
        "2022_sdg": [],
        "2023_sdg": [],
    }

    MAIN_DIR = os.path.join("results", "retrieve")

    for company in os.listdir(MAIN_DIR):
        company_dir = os.path.join(MAIN_DIR, company)
        if os.path.isdir(company_dir):
            results_path = os.path.join(company_dir, "results.json")
            results_ai_path = os.path.join(company_dir, "results_ai.json")
            scores = get_scores(results_path)
            ai_scores = get_ai_scores(results_ai_path)

            years = list(range(2014,2024))
            company_name = company.split(".")[1].split("_")[0]
            df["company"].append(company_name)
            for year in years:
                sdg_year = str(year) + "_sdg"
                if str(year) in list(scores.keys()):
                    df[sdg_year].append(scores[str(year)][0])
                else:
                    df[sdg_year].append("NA")

                ai_year = str(year) + "_ai"
                if str(year) in list(ai_scores.keys()):
                    df[ai_year].append(ai_scores[str(year)])
                else:
                    df[ai_year].append("NA")

    df = pd.DataFrame.from_dict(df)
    df.to_csv("results/retrieve/results.csv")


if __name__ == "__main__":
    # scores = get_scores(pth)
    # print(scores)
    get_sdg_scores()
    # get_ai_scores(pth)