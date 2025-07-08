import os

base_dir = os.path.join('data', 'texts')

files = []
for dirname, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith('rag_filter_ai.json') or filename.endswith('rag_filter_sdg.json'):
            files.append(os.path.join(dirname, filename))

print(f"Found {len(files)} files to remove!")

for file in files:
    # print(file)
    os.remove(file)