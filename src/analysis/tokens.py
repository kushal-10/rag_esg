import tiktoken
import glob
import json
from tqdm import tqdm

encoding = tiktoken.get_encoding("gpt2")  # Or "cl100k_base" for GPT-4/3.5

json_files = glob.glob('data/texts/**/splits.json', recursive=True)
total_tokens = 0

for file in tqdm(json_files):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sentence in data.values():
            num_tokens = len(encoding.encode(sentence))
            total_tokens += num_tokens

print("Total number of tokens (tiktoken/gpt2):", total_tokens)
