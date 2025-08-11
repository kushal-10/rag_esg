import os

base_dirs = ["data/texts", "data/textsv2", "data/textsv3", "data/reports_corrupted"]

counter = 0
blanks = 0
for base_dir in base_dirs:
    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith("results.txt"):
                counter += 1
                with open(os.path.join(dirpath, filename), "r") as f:
                    text = f.read()
                if not text:
                    blanks += 1
                    print(f"{counter}: {filename, dirpath}")


print(counter, blanks)

