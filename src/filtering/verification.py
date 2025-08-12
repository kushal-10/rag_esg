import json
import os

import pandas as pd
import tiktoken
from tqdm import tqdm

enc = tiktoken.encoding_for_model("gpt-4o-mini")


csv_path = "data/scores_csv/14.basf_$42.93 b_industrials/2023/similarity_scores.csv"
json_path = "data/texts/14.basf_$42.93 b_industrials/2023/splits.json"

def get_tokens(csv_path, json_path):

    with open(json_path) as json_file:
        data = json.load(json_file)

    df = pd.read_csv(csv_path)

    T = 0.4


    cols = list(df.columns)
    tokens = 0
    sentences = 0
    for i in range(len(df)):
        for j in range(1, len(cols)):
            score = df.iloc[i][cols[j]]

            if score >= T:
                sent_id = int(df.iloc[i]["sentence_id"])
                sentence = data[str(sent_id)]
                sentences += 1
                toks = len(enc.encode(str(sentence)))
                tokens += toks
                break

    return tokens, sentences

if __name__ == "__main__":

    csvs = []
    base_dir = "data/scores_csv"
    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                filepath = os.path.join(dirpath, filename)
                csvs.append(filepath)

    total_toks = 0
    total_sentences = 0
    for i in tqdm(range(len(csvs))):
        csv_path = csvs[i]
        json_path = csv_path.replace("similarity_scores.csv", "splits.json")
        json_path = json_path.replace("scores_csv", "texts")
        tokens, sents = get_tokens(csv_path, json_path)
        total_toks += tokens
        total_sentences += sents

        print(total_toks, total_sentences)

    # Toks, Sents for T=0.4 est on 50 reports
    # -> 628321, 160774 -> 161k sents, 630k tokens, so for 1410 report it is:
    # Tokens - ~20M tokens
    # Sentences - ~500k sentences -> ~60M tokens for a 120 token base prompt -> 80M tokens to process
    # Output -> 10M tokens
    # Nano - IP cost - $0.4, OP Cost - $4
    # ~5$ total cost

