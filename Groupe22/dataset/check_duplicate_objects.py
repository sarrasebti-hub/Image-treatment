import json
from collections import Counter

# Load metadata
path = "dataset/dataset/metadata/knowledge_base_metadata.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

total_objects = 0
duplicate_objects = 0
duplicate_details = []

for image in data:
    objects = image["objects"]
    total_objects += len(objects)

    # Identify identical object annotations
    seen = Counter(
        (
            obj["category"],
            tuple(obj["bbox"]),
            obj["area"],
            obj["iscrowd"]
        )
        for obj in objects
    )

    for obj, count in seen.items():
        if count > 1:
            duplicate_objects += count - 1

            if len(duplicate_details) < 10:
                duplicate_details.append({
                    "image_id": image["image_id"],
                    "object": obj,
                    "occurrences": count
                })

print("\n========== DUPLICATE OBJECT AUDIT ==========")
print("Total objects:", total_objects)
print("Duplicate object annotations:", duplicate_objects)
print("Examples:")

for item in duplicate_details:
    print(item)

print("\n========== AUDIT COMPLETED ==========")