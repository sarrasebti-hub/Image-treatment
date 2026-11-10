
import json
import csv
import os
from collections import Counter

# ==========================================
# TOP 50 OBJECT CLASSES - KNOWLEDGE BASE
# ==========================================

METADATA_PATH = (
    "dataset/dataset/metadata/"
    "knowledge_base_metadata.json"
)

OUTPUT_DIR = "results/statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Knowledge Base metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) == 10000, "Expected exactly 10,000 images."

object_counts = Counter()
image_counts = Counter()

for image in data:
    image_id = image["image_id"]

    for obj in image.get("objects", []):
        category = obj["category"]
        object_counts[category] += 1
        image_counts[category] += 1

# Select the 50 most frequent classes
top_50 = object_counts.most_common(50)

print("\n========== TOP 50 OBJECT CLASSES ==========")

for rank, (category, count) in enumerate(top_50, start=1):
    print(f"{rank}. {category}: {count} objects")

print("\nTotal images:", len(data))
print("Total objects:", sum(object_counts.values()))
print("Unique categories:", len(object_counts))

# Save to a new CSV file
csv_path = os.path.join(
    OUTPUT_DIR,
    "top_50_object_classes_corrected.csv"
)

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Rank",
        "Object Class",
        "Number of Objects"
    ])

    for rank, (category, count) in enumerate(top_50, start=1):
        writer.writerow([
            rank,
            category,
            count
        ])

print("\nCSV saved:", csv_path)

# Validation
assert sum(object_counts.values()) == 72754
assert len(object_counts) == 80
assert len(top_50) == 50

print("\nVALIDATION PASSED")
print("Top 50 analysis completed successfully.")
