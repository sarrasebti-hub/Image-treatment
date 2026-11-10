import json
import re

BASE = "dataset/dataset"

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

# Knowledge Base
kb = load(f"{BASE}/metadata/knowledge_base_metadata.json")

train_ids = {int(x["image_id"]) for x in kb}
train_captions = {
    normalize(c)
    for x in kb
    for c in x["captions"]
}

# Validation
val = load(f"{BASE}/splits/validation/validation_cleaned.json")
val_ids = {int(x["id"]) for x in val["images"]}
val_captions = {
    normalize(x["caption"])
    for x in val["annotations"]
}

# Test
test = load(f"{BASE}/splits/test/test_cleaned.json")
test_ids = {int(x["id"]) for x in test["images"]}
test_captions = {
    normalize(x["caption"])
    for x in test["annotations"]
}

print("========== IMAGE SPLIT LEAKAGE ==========")
print("Train images:", len(train_ids))
print("Validation images:", len(val_ids))
print("Test images:", len(test_ids))

print("Train / Validation overlap:", len(train_ids & val_ids))
print("Train / Test overlap:", len(train_ids & test_ids))
print("Validation / Test overlap:", len(val_ids & test_ids))

print("\n========== CAPTION LEAKAGE ==========")
print("Train / Validation exact overlap:",
      len(train_captions & val_captions))
print("Train / Test exact overlap:",
      len(train_captions & test_captions))
print("Validation / Test exact overlap:",
      len(val_captions & test_captions))

print("\n========== TEST EXCLUSION ==========")
print("Test IDs inside Knowledge Base:", len(test_ids & train_ids))
print("Test captions inside Knowledge Base:",
      len(test_captions & train_captions))