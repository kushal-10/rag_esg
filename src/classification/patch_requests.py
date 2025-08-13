import os, json, shutil

from tqdm import tqdm

BATCH_DIR = os.path.join("data", "batches_41_mini")
PATCHED_DIR = os.path.join(BATCH_DIR, "patched_max_tokens_50")
os.makedirs(PATCHED_DIR, exist_ok=True)

for fname in tqdm(os.listdir(BATCH_DIR)):
    if not fname.endswith(".jsonl"):
        continue
    src = os.path.join(BATCH_DIR, fname)
    dst = os.path.join(PATCHED_DIR, fname)

    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            body = obj.setdefault("body", {})
            # inject or overwrite
            body["max_tokens"] = 50
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Patched files written to: {PATCHED_DIR}")
