import os

BASE_DIR = os.path.join("data", "cleaned_reports")

total_pdfs = 0
for dirname, _, filenames in os.walk(BASE_DIR):
    for filename in filenames:
        if filename.endswith(".pdf"):
            total_pdfs += 1


TXT_DIR = os.path.join("data", "texts")

total_txts = 0
for dirname, _, filenames in os.walk(TXT_DIR):
    for filename in filenames:
        if filename.endswith(".txt"):
            total_txts += 1

print(f"Total pdfs: {total_pdfs}")
print(f"Total txts: {total_txts}")
