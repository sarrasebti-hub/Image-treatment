
import json
import random
from collections import Counter
from pathlib import Path

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ANNOTATIONS_DIR = Path("data/raw/annotations/annotations")
OUTPUT_DIR = Path("data/processed/splits")

TRAIN_INSTANCES = ANNOTATIONS_DIR / "instances_train2017.json"
VAL_INSTANCES = ANNOTATIONS_DIR / "instances_val2017.json"

TRAIN_CAPTIONS = ANNOTATIONS_DIR / "captions_train2017.json"
VAL_CAPTIONS = ANNOTATIONS_DIR / "captions_val2017.json"

SEED = 42

# --------------------------------------------------
# 2. Load JSON files
# --------------------------------------------------

def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


train_instances = load_json(TRAIN_INSTANCES)
val_instances = load_json(VAL_INSTANCES)

train_captions = load_json(TRAIN_CAPTIONS)
val_captions = load_json(VAL_CAPTIONS)

# --------------------------------------------------
# 3. Select Knowledge Base and Validation IDs
# --------------------------------------------------

random.seed(SEED)

train_ids = [
    image["id"] for image in train_instances["images"]
]

random.shuffle(train_ids)

knowledge_base_ids = train_ids[:10000]
validation_ids = train_ids[10000:10100]

# --------------------------------------------------
# 4. Prepare object information for Test
# --------------------------------------------------

object_counts = Counter()
categories_per_image = {}

for annotation in val_instances["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    object_counts[image_id] += 1

    if image_id not in categories_per_image:
        categories_per_image[image_id] = set()

    categories_per_image[image_id].add(category_id)

# Include images with zero annotated objects
for image in val_instances["images"]:
    image_id = image["id"]
    object_counts.setdefault(image_id, 0)
    categories_per_image.setdefault(image_id, set())

# --------------------------------------------------
# 5. Stratify Test images by object count
# --------------------------------------------------

def scene_group(count):
    if count <= 3:
        return "simple"
    elif count <= 10:
        return "medium"
    else:
        return "complex"


groups = {
    "simple": [],
    "medium": [],
    "complex": []
}

for image_id, count in object_counts.items():
    groups[scene_group(count)].append(image_id)

# Allocate 100 images proportionally
total_images = sum(len(group) for group in groups.values())

quotas = {
    name: round(len(group) / total_images * 100)
    for name, group in groups.items()
}

# Correct rounding to ensure exactly 100
while sum(quotas.values()) < 100:
    name = max(
        groups,
        key=lambda key: len(groups[key]) / total_images * 100
        - quotas[key]
    )
    quotas[name] += 1

while sum(quotas.values()) > 100:
    name = max(quotas, key=quotas.get)
    quotas[name] -= 1

test_ids = []

for name, group in groups.items():
    test_ids.extend(
        random.sample(group, quotas[name])
    )

random.shuffle(test_ids)

# --------------------------------------------------
# 6. Check split integrity
# --------------------------------------------------

assert len(knowledge_base_ids) == 10000
assert len(validation_ids) == 100
assert len(test_ids) == 100

assert set(knowledge_base_ids).isdisjoint(validation_ids)

assert set(knowledge_base_ids).isdisjoint(test_ids)
assert set(validation_ids).isdisjoint(test_ids)

# --------------------------------------------------
# 7. Save split IDs
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

splits = {
    "knowledge_base": knowledge_base_ids,
    "validation": validation_ids,
    "test": test_ids
}

for name, ids in splits.items():
    output_file = OUTPUT_DIR / f"{name}_ids.json"

    with open(output_file, "w") as file:
        json.dump(ids, file, indent=4)

# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\n========== SPLIT SUMMARY ==========")

print("Knowledge Base:", len(knowledge_base_ids))
print("Validation:", len(validation_ids))
print("Test:", len(test_ids))

print("\nTest scene distribution:")

for name, group in groups.items():
    selected = [
        image_id for image_id in test_ids
        if image_id in set(group)
    ]

    print(f"{name}: {len(selected)} images")

print("\nTotal selected images:", 
      len(knowledge_base_ids) + len(validation_ids) + len(test_ids))

print("\nAll split checks passed!")
print("Split IDs saved in:", OUTPUT_DIR)
