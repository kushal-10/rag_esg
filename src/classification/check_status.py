import json
from openai import OpenAI

BATCH_LOG = "batch_ids.json"  # path to your batch log
client = OpenAI()

with open(BATCH_LOG, "r", encoding="utf-8") as f:
    entries = json.load(f)

for e in entries:
    batch_id = e["batch_id"]
    try:
        b = client.batches.retrieve(batch_id)
        print(f"{batch_id}: status={b.status}, output_file_id={getattr(b, 'output_file_id', None)}")
    except Exception as ex:
        print(f"{batch_id}: ERROR retrieving status -> {ex}")
