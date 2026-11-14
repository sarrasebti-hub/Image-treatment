
import json
from pathlib import Path
from collections import Counter
from PIL import Image

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

TRAIN_DIR = DATASET_DIR / "train2017"
VAL_DIR = DATASET_DIR / "val2017"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REPORT = {}


# ==========================================
# 2. LOAD COCO ANNOTATIONS
# ==========================================

def load_json(path):
    print(f"Loading: {path.name}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


train_captions = load_json(
    ANNOTATIONS_DIR / "captions_train2017.json"
)

val_captions = load_json(
    ANNOTATIONS_DIR / "captions_val2017.json"
)

train_instances = load_json(
    ANNOTATIONS_DIR / "instances_train2017.json"
)

val_instances = load_json(
    ANNOTATIONS_DIR / "instances_val2017.json"
)


# ==========================================
# 3. CAPTION AND IMAGE AUDIT
# ==========================================

def audit_captions(data, image_dir, dataset_name):

    images = data["images"]
    annotations = data["annotations"]

    image_ids = [image["id"] for image in images]
    known_ids = set(image_ids)

    caption_counts = Counter(
        annotation["image_id"]
        for annotation in annotations
    )

    duplicate_ids = [
        image_id
        for image_id, count in Counter(image_ids).items()
        if count > 1
    ]

    unknown_caption_ids = [
        annotation["image_id"]
        for annotation in annotations
        if annotation["image_id"] not in known_ids
    ]

    missing = []
    unreadable = []

    print(f"\n========== {dataset_name} ==========")

    print("Images in annotation file:", len(images))
    print("Unique image IDs:", len(known_ids))
    print("Duplicate image IDs:", len(duplicate_ids))
    print("Total captions:", len(annotations))

    # Captions per image
    distribution = Counter(caption_counts.values())

    print("\nCaptions per image distribution:")

    for number, count in sorted(distribution.items()):
        print(f"{number} captions: {count} images")

    incorrect = [
        image_id
        for image_id in image_ids
        if caption_counts[image_id] != 5
    ]

    print("Images without exactly 5 captions:", len(incorrect))

    # Check missing and unreadable images
    print("\nChecking image files...")

    for index, image in enumerate(images, start=1):

        image_path = image_dir / image["file_name"]

        if not image_path.exists():
            missing.append(image["file_name"])
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()

        except Exception:
            unreadable.append(image["file_name"])

        if index % 10000 == 0:
            print(f"Checked {index}/{len(images)} images")

    print("Missing image files:", len(missing))
    print("Unreadable/corrupted images:", len(unreadable))
    print("Caption references to unknown image IDs:",
          len(unknown_caption_ids))

    result = {
        "images_in_annotations": len(images),
        "unique_image_ids": len(known_ids),
        "duplicate_image_ids": duplicate_ids,
        "total_captions": len(annotations),
        "captions_per_image_distribution": {
            str(k): v for k, v in sorted(distribution.items())
        },
        "images_without_exactly_5_captions": incorrect,
        "missing_images": missing,
        "unreadable_images": unreadable,
        "unknown_caption_image_ids": unknown_caption_ids
    }

    return known_ids, result


train_ids, train_report = audit_captions(
    train_captions, TRAIN_DIR, "TRAIN2017"
)

val_ids, val_report = audit_captions(
    val_captions, VAL_DIR, "VAL2017"
)


# ==========================================
# 4. OBJECT ANNOTATION AUDIT
# ==========================================

def audit_instances(data, known_ids, dataset_name):

    annotations = data["annotations"]
    categories = data["categories"]

    category_names = {
        category["id"]: category["name"]
        for category in categories
    }

    object_counts = Counter(
        annotation["category_id"]
        for annotation in annotations
    )

    unknown_image_ids = [
        annotation["image_id"]
        for annotation in annotations
        if annotation["image_id"] not in known_ids
    ]

    unknown_category_ids = [
        annotation["category_id"]
        for annotation in annotations
        if annotation["category_id"] not in category_names
    ]

    print(f"\n========== {dataset_name} OBJECTS ==========")

    print("Total object annotations:", len(annotations))
    print("Number of object categories:", len(categories))
    print("Object references to unknown image IDs:",
          len(unknown_image_ids))
    print("Unknown category references:", len(unknown_category_ids))

    print("\nTop 10 most frequent object categories:")

    for category_id, count in object_counts.most_common(10):
        print(f"{category_names.get(category_id, 'Unknown')}: {count}")

    return {
        "total_object_annotations": len(annotations),
        "number_of_categories": len(categories),
        "object_references_to_unknown_image_ids": unknown_image_ids,
        "unknown_category_ids": unknown_category_ids,
        "top_10_categories": {
            category_names.get(k, "Unknown"): v
            for k, v in object_counts.most_common(10)
        }
    }


train_objects_report = audit_instances(
    train_instances, train_ids, "TRAIN2017"
)

val_objects_report = audit_instances(
    val_instances, val_ids, "VAL2017"
)


# ==========================================
# 5. SAVE AUDIT REPORT
# ==========================================

REPORT = {
    "train2017": {
        "captions": train_report,
        "objects": train_objects_report
    },
    "val2017": {
        "captions": val_report,
        "objects": val_objects_report
    }
}

report_path = RESULTS_DIR / "dataset_audit_report.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(REPORT, file, indent=4)

print("\n========== AUDIT FINISHED ==========")
print("Report saved to:", report_path)
