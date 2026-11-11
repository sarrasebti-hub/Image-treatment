import json
from collections import Counter
from pathlib import Path

# Project paths
ANNOTATIONS_PATH = Path("data/raw/annotations/annotations/instances_train2017.json")
SPLIT_PATH = Path("data/processed/splits/knowledge_base_ids.json")

# Load COCO annotations
with open(ANNOTATIONS_PATH, "r") as file:
    coco_data = json.load(file)

# Load Knowledge Base image IDs
with open(SPLIT_PATH, "r") as file:
    knowledge_base_ids = set(json.load(file))

# Map category IDs to category names
categories = {
    category["id"]: category["name"]
    for category in coco_data["categories"]
}

# Count objects in Knowledge Base images
category_counts = Counter()

for annotation in coco_data["annotations"]:
    if annotation["image_id"] in knowledge_base_ids:
        category_name = categories[annotation["category_id"]]
        category_counts[category_name] += 1

# Display results
print("Object category distribution in Knowledge Base:\n")

for category, count in category_counts.most_common():
    print(f"{category}: {count}")

print("\nTotal objects:", sum(category_counts.values()))
print("Number of categories:", len(category_counts))
