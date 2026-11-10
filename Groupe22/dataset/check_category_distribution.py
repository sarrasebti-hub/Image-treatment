import json
from collections import Counter

# Load metadata
path = "dataset/dataset/metadata/knowledge_base_metadata.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

category_counts = Counter()
images_per_category = Counter()

total_objects = 0

for image in data:
    categories_in_image = set()

    for obj in image["objects"]:
        category = obj["category"]

        category_counts[category] += 1
        categories_in_image.add(category)
        total_objects += 1

    for category in categories_in_image:
        images_per_category[category] += 1

print("\n========== COCO CATEGORY DISTRIBUTION ==========")

print("Total images:", len(data))
print("Total objects:", total_objects)
print("Unique categories:", len(category_counts))

print("\n========== TOP 50 OBJECT CLASSES ==========")

for rank, (category, count) in enumerate(
    category_counts.most_common(50), start=1
):
    print(
        f"{rank}. {category}: "
        f"{count} objects in "
        f"{images_per_category[category]} images"
    )

print("\n========== CATEGORY AUDIT ==========")

print("Categories with at least one object:", len(category_counts))
print("Categories without objects:", 80 - len(category_counts))

print("\n========== AUDIT COMPLETED ==========")