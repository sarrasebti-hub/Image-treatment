import os
import json
import random

# =====================================
# COCO 2017 - DATA SPLITS
# Groupe 22 - Corrective RAG
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "train2017")
VAL_DIR = os.path.join(BASE_DIR, "val2017")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
OUTPUT_DIR = os.path.join(BASE_DIR, "splits")

KB_SIZE = 10000
VALIDATION_SIZE = 100
TEST_SIZE = 100
SEED = 42

random.seed(SEED)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_local_images(folder):
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(folder)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    }


def prepare_split(image_ids, data):
    selected_ids = set(image_ids)

    images = [
        img for img in data["images"]
        if str(img["id"]) in selected_ids
    ]

    captions_by_image = {}

    for ann in data["annotations"]:
        image_id = str(ann["image_id"])

        if image_id in selected_ids:
            captions_by_image.setdefault(image_id, []).append(ann)

    captions = []

    for image_id in image_ids:
        image_captions = captions_by_image.get(image_id, [])

        if len(image_captions) < 5:
            raise ValueError(
                f"Image {image_id} has fewer than 5 captions"
            )

        captions.extend(image_captions[:5])

    return {
        "images": images,
        "annotations": captions
    }


def save_split(name, data):
    folder = os.path.join(OUTPUT_DIR, name)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{name}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"{name}: {len(data['images'])} images, "
        f"{len(data['annotations'])} captions"
    )


# =====================================
# LOAD DATA
# =====================================

print("Loading COCO annotations...")

train_data = load_json(
    os.path.join(ANNOTATIONS_DIR, "captions_train2017.json")
)

val_data = load_json(
    os.path.join(ANNOTATIONS_DIR, "captions_val2017.json")
)


# =====================================
# SELECT LOCAL IMAGES
# =====================================

train_local = get_local_images(TRAIN_DIR)
val_local = get_local_images(VAL_DIR)

train_available = [
    str(img["id"])
    for img in train_data["images"]
    if img["file_name"].rsplit(".", 1)[0] in train_local
]

val_available = [
    str(img["id"])
    for img in val_data["images"]
    if img["file_name"].rsplit(".", 1)[0] in val_local
]

print("Available train images:", len(train_available))
print("Available validation images:", len(val_available))

if len(train_available) < KB_SIZE:
    raise ValueError("Not enough local training images.")

if len(val_available) < VALIDATION_SIZE + TEST_SIZE:
    raise ValueError("Not enough local validation images.")


# =====================================
# CREATE SPLITS
# =====================================

random.shuffle(train_available)
random.shuffle(val_available)

kb_ids = train_available[:KB_SIZE]

validation_ids = val_available[:VALIDATION_SIZE]

test_ids = val_available[
    VALIDATION_SIZE:VALIDATION_SIZE + TEST_SIZE
]


# =====================================
# PREPARE AND SAVE
# =====================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

save_split(
    "knowledge_base",
    prepare_split(kb_ids, train_data)
)

save_split(
    "validation",
    prepare_split(validation_ids, val_data)
)

save_split(
    "test",
    prepare_split(test_ids, val_data)
)


# =====================================
# DATA LEAKAGE CHECK
# =====================================

kb_set = set(kb_ids)
validation_set = set(validation_ids)
test_set = set(test_ids)

assert kb_set.isdisjoint(validation_set)
assert kb_set.isdisjoint(test_set)
assert validation_set.isdisjoint(test_set)

print("\n===== DATA LEAKAGE CHECK =====")
print("Knowledge Base / Validation:", len(kb_set & validation_set))
print("Knowledge Base / Test:", len(kb_set & test_set))
print("Validation / Test:", len(validation_set & test_set))
print("No image overlap detected.")
print("==============================")