import json
import os

base_dir = "data/texts"

no_json = 0
empty_json = 0
total = 0 # 1410 text files
for dirname, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith(".txt"):
            filepath = os.path.join(dirname, filename)
            if "results.txt" not in filepath:
                print(filepath)
            else:
                total += 1
print(total)





