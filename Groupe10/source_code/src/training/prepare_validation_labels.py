import json
from pathlib import Path
from collections import defaultdict

ANNOTATIONS_PATH = Path(
    "data/raw/annotations/annotations/instances_train2017.json"
)

IMAGES_DIR = Path("data/processed/validation/images")
LABELS_DIR = Path("data/processed/yolo/labels/val")

LABELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading COCO annotations...")

with open(ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

# Map COCO category IDs to YOLO class IDs (0 to 79)
categories = sorted(coco["categories"], key=lambda x: x["id"])

category_mapping = {
    category["id"]: index
    for index, category in enumerate(categories)
}

# Map image IDs to image information
images = {
    image["id"]: image
    for image in coco["images"]
}

# Keep only images belonging to our Validation split
validation_images = {
    int(path.stem): path
    for path in IMAGES_DIR.glob("*.jpg")
}

# Group annotations by image ID
annotations_by_image = defaultdict(list)

for annotation in coco["annotations"]:
    image_id = annotation["image_id"]

    if image_id in validation_images:
        annotations_by_image[image_id].append(annotation)

print(f"Validation images: {len(validation_images)}")

created = 0
empty = 0
total_boxes = 0

for image_id, image_path in validation_images.items():

    image_info = images[image_id]

    image_width = image_info["width"]
    image_height = image_info["height"]

    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    lines = []

    for annotation in annotations_by_image[image_id]:

        category_id = annotation["category_id"]
        class_id = category_mapping[category_id]

        x, y, width, height = annotation["bbox"]

        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height

        normalized_width = width / image_width
        normalized_height = height / image_height

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{normalized_width:.6f} "
            f"{normalized_height:.6f}"
        )

    label_path.write_text("\n".join(lines))

    created += 1
    total_boxes += len(lines)

    if not lines:
        empty += 1

print("\n===== VALIDATION YOLO LABELS =====")
print(f"Label files created: {created}")
print(f"Images without annotations: {empty}")
print(f"Total bounding boxes: {total_boxes}")
print(f"Labels directory: {LABELS_DIR}")