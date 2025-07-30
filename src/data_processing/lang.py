import os
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # for reproducibility

results_dir = "data/texts"
german_files = []

for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith("results.txt"):
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                # Read a sample of the file (e.g., first 1000 chars)
                text = f.read(1000)
                try:
                    lang = detect(text)
                    if lang == "de":
                        print(f"German detected: {file_path}")
                        german_files.append(file_path)
                except Exception as e:
                    print(f"Could not detect language for {file_path}: {e}")

print("Total German files:", len(german_files))
