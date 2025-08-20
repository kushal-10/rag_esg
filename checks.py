import os

base_dir = os.path.join("data", "texts")

lists = os.listdir(base_dir)
sorted = sorted(lists)
print(sorted)