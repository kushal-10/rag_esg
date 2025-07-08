import os

base_dir = os.path.join('data', 'texts')

files = []
for dirname, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith('sentence_scores.csv'):
            files.append(os.path.join(dirname, filename))

print(f"Found {len(files)} files to remove!")

for file in files:
    print(file)
    # os.remove(file)