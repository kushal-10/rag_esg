# cancel_and_fetch.py
import time, json, os
from typing import List
from openai import OpenAI

client = OpenAI()

BATCH_IDS: List[str] = [
    "batch_689b6aa509ec8190bd6d681ce14df250",
    "batch_689b6c10888c8190981da73c594e8153",
    "batch_689b6c1d675881908689407beb81b364",
    "batch_689b6c2d6668819086c4180630906ddf",
    "batch_689b6c39924081909128466200e9374b",
    "batch_689b6c49869c8190a9e954d9d3966e7b",
    "batch_689b6c685cb08190b457c74cb51e3838",
    "batch_689b6c7d8f588190895fe321f4f52be1",
    "batch_689b6c8b44bc819090e16b5131b73403",
    "batch_689b6c96f31c81908bb8ab4d96efc575",
    "batch_689b6ca3a46c8190a21a8ada85421a3f",
]

OUT_DIR = "batch_results"
os.makedirs(OUT_DIR, exist_ok=True)

TERMINAL = {"completed", "failed", "cancelled", "expired"}
STILL_RUNNING = {"validating", "in_progress", "finalizing", "cancelling"}

def save_file(file_id: str, path: str):
    # New SDK: content() returns a stream-like object with helpers
    content = client.files.content(file_id)
    # Try best-effort to write regardless of underlying transport
    if hasattr(content, "write_to_file"):
        content.write_to_file(path)
    elif hasattr(content, "text"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.text)
    else:
        # Fallback: bytes from .content or .read()
        data = getattr(content, "content", None)
        if data is None and hasattr(content, "read"):
            data = content.read()
        mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
        with open(path, mode) as f:
            f.write(data)

def cancel_if_needed(bid: str):
    try:
        bj = client.batches.retrieve(bid)
        if bj.status in {"validating", "in_progress", "finalizing"}:
            bj = client.batches.cancel(bid)
            print(f"[{bid}] sent cancel → status {bj.status}")
        else:
            print(f"[{bid}] skip cancel, status {bj.status}")
    except Exception as e:
        print(f"[{bid}] cancel error: {e}")

def poll_until_terminal(bid: str, timeout_s=900, interval_s=10):
    t0 = time.time()
    while True:
        bj = client.batches.retrieve(bid)
        counts = getattr(bj, "request_counts", None)
        counts_str = f" | counts={counts}" if counts else ""
        print(f"[{bid}] status={bj.status}{counts_str}")
        if bj.status in TERMINAL:
            return bj
        if time.time() - t0 > timeout_s:
            print(f"[{bid}] timeout while waiting; continuing anyway.")
            return bj
        time.sleep(interval_s)

def download_outputs(bj, bid: str):
    out_id = getattr(bj, "output_file_id", None)
    err_id = getattr(bj, "error_file_id", None)

    if out_id:
        out_path = os.path.join(OUT_DIR, f"{bid}_output.jsonl")
        save_file(out_id, out_path)
        print(f"[{bid}] saved output → {out_path}")
    else:
        print(f"[{bid}] no output_file_id")

    if err_id:
        err_path = os.path.join(OUT_DIR, f"{bid}_errors.jsonl")
        save_file(err_id, err_path)
        print(f"[{bid}] saved errors → {err_path}")
    else:
        print(f"[{bid}] no error_file_id")

def main():
    # 1) Cancel everything still running
    for bid in BATCH_IDS:
        cancel_if_needed(bid)

    # 2) Poll each to terminal and 3) download files if present
    for bid in BATCH_IDS:
        bj = poll_until_terminal(bid)
        download_outputs(bj, bid)

if __name__ == "__main__":
    main()
