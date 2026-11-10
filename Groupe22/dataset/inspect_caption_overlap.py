import json
import re
from collections import defaultdict

BASE = "dataset/dataset"

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

kb = load(f"{BASE}/metadata/knowledge_base_metadata.json")
test = load(f"{BASE}/splits/test/test_cleaned.json")

train_map = defaultdict(list)
test_map = defaultdict(list)

for item in kb:
    for caption in item["captions"]:
        train_map[normalize(caption)].append(int(item["image_id"]))

for item in test["annotations"]:
    test_map[normalize(item["caption"])].append(int(item["image_id"]))

common = set(train_map) & set(test_map)

print("========== EXACT CAPTION OVERLAP DETAILS ==========")
print("Number of shared captions:", len(common))

for i, caption in enumerate(sorted(common), 1):
    print(f"\n{i}. Caption: {caption}")
    print("Train image IDs:", train_map[caption])
    print("Test image IDs:", test_map[caption])
    print(
        "Different image IDs:",
        not bool(set(train_map[caption]) & set(test_map[caption]))
    )