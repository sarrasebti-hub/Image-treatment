
import json
import random
from pathlib import Path
from collections import Counter

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_DIR = PROJECT_ROOT / "datasets" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "splits"

SEED = 42

KB_SIZE = 10_000
VALIDATION_SIZE = 100
TEST_SIZE = 1_000


# ==========================================
# 2. LOAD ANNOTATIONS
# ==========================================

def load_json(filename):
    path = ANNOTATIONS_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


print("Loading COCO annotations...")

train_captions = load_json("captions_train2017.json")
val_captions = load_json("captions_val2017.json")
val_instances = load_json("instances_val2017.json")


# ==========================================
# 3. EXTRACT IMAGE IDs
# ==========================================

train_ids = sorted(
    image["id"] for image in train_captions["images"]
)

test_pool_ids = sorted(
    image["id"] for image in val_captions["images"]
)

print("TRAIN2017 images:", len(train_ids))
print("VAL2017 images:", len(test_pool_ids))


# ==========================================
# 4. CREATE KNOWLEDGE BASE AND VALIDATION
# ==========================================

rng = random.Random(SEED)

selected_train = rng.sample(
    train_ids,
    KB_SIZE + VALIDATION_SIZE
)

knowledge_base_ids = selected_train[:KB_SIZE]
validation_ids = selected_train[KB_SIZE:]


# ==========================================
# 5. CALCULATE OBJECT COUNTS
# ==========================================

object_counts = Counter(
    annotation["image_id"]
    for annotation in val_instances["annotations"]
)

# Include images that have zero annotated objects.
for image_id in test_pool_ids:
    object_counts.setdefault(image_id, 0)


# ==========================================
# 6. STRATIFY TEST IMAGES
# ==========================================

sorted_counts = sorted(
    object_counts[image_id]
    for image_id in test_pool_ids
)

quartiles = [
    sorted_counts[int(len(sorted_counts) * 0.25)],
    sorted_counts[int(len(sorted_counts) * 0.50)],
    sorted_counts[int(len(sorted_counts) * 0.75)]
]

groups = {
    "simple": [],
    "medium": [],
    "complex": [],
    "very_complex": []
}

for image_id in test_pool_ids:

    count = object_counts[image_id]

    if count <= quartiles[0]:
        groups["simple"].append(image_id)

    elif count <= quartiles[1]:
        groups["medium"].append(image_id)

    elif count <= quartiles[2]:
        groups["complex"].append(image_id)

    else:
        groups["very_complex"].append(image_id)


# ==========================================
# 7. SELECT 1,000 TEST IMAGES
# ==========================================

test_rng = random.Random(SEED)

test_ids = []

base_size = TEST_SIZE // 4
remaining = TEST_SIZE

for index, (group_name, group_ids) in enumerate(groups.items()):

    if index == 3:
        amount = remaining
    else:
        amount = base_size
        remaining -= amount

    if len(group_ids) < amount:
        raise ValueError(
            f"Not enough images in group: {group_name}"
        )

    test_ids.extend(
        test_rng.sample(group_ids, amount)
    )

test_ids.sort()


# ==========================================
# 8. VALIDATE SPLITS
# ==========================================

assert len(knowledge_base_ids) == KB_SIZE
assert len(validation_ids) == VALIDATION_SIZE
assert len(test_ids) == TEST_SIZE

assert set(knowledge_base_ids).isdisjoint(validation_ids)
assert set(knowledge_base_ids).isdisjoint(test_ids)
assert set(validation_ids).isdisjoint(test_ids)

assert set(knowledge_base_ids).issubset(set(train_ids))
assert set(validation_ids).issubset(set(train_ids))
assert set(test_ids).issubset(set(test_pool_ids))

print("\nAll split integrity checks passed!")


# ==========================================
# 9. SAVE SPLIT FILES
# ==========================================

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

def save_json(filename, data):

    path = SPLITS_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print("Saved:", path.name)


save_json("knowledge_base_ids.json", knowledge_base_ids)
save_json("validation_ids.json", validation_ids)
save_json("test_ids.json", test_ids)


# ==========================================
# 10. SAVE SPLIT METADATA
# ==========================================

metadata = {
    "random_seed": SEED,
    "knowledge_base_size": KB_SIZE,
    "validation_size": VALIDATION_SIZE,
    "test_size": TEST_SIZE,
    "test_stratification": "Object-count quartiles",
    "object_count_boundaries": quartiles,
    "test_group_sizes": {
        name: len(ids) for name, ids in groups.items()
    },
    "selected_test_group_sizes": {
        name: sum(
            image_id in set(test_ids)
            for image_id in ids
        )
        for name, ids in groups.items()
    }
}

save_json("split_metadata.json", metadata)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n========== SPLIT CREATION COMPLETE ==========")

print("Knowledge Base:", len(knowledge_base_ids))
print("Validation:", len(validation_ids))
print("Testing:", len(test_ids))

print("\nTest selection by scene complexity:")

for name, amount in metadata["selected_test_group_sizes"].items():
    print(f"{name}: {amount}")

print("\nRandom seed:", SEED)
print("All files saved successfully.")
