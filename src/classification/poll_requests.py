import os
from openai import OpenAI

client = OpenAI()

SAVE_DIR = os.path.join("data", "batch_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# List batches (adjust limit if you have more)
batches = client.batches.list(limit=100)

for b in batches.data:
    if b.status == "completed" and b.output_file_id:
        print(f"Downloading results for {b.id} ...")

        # Retrieve the file content
        content = client.files.content(b.output_file_id).read().decode("utf-8")

        # Save locally
        save_path = os.path.join(SAVE_DIR, f"{b.id}.jsonl")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Saved: {save_path}")
