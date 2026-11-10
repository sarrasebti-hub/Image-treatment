import json
from pathlib import Path
from collections import Counter

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

METADATA_PATH = BASE_DIR / "metadata" / "knowledge_base_metadata.json"
INSTANCES_PATH = BASE_DIR.parent / "annotations" / "instances_train2017.json"

# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

print("Chargement des fichiers...")

with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

with open(INSTANCES_PATH, encoding="utf-8") as f:
    instances = json.load(f)

valid_categories = {
    category["name"] for category in instances["categories"]
}

# ==========================================
# 3. INITIALISATION DES CONTROLES
# ==========================================

unknown_categories = Counter()
invalid_bboxes = []
invalid_areas = []
invalid_dimensions = []
missing_fields = []
duplicate_image_ids = []
image_ids = [item["image_id"] for item in metadata]

# ==========================================
# 4. VERIFICATION DES DOUBLONS
# ==========================================

seen = set()

for image_id in image_ids:
    if image_id in seen:
        duplicate_image_ids.append(image_id)
    seen.add(image_id)

# ==========================================
# 5. VERIFICATION DES IMAGES ET OBJETS
# ==========================================

for item in metadata:

    image_id = item["image_id"]

    # Dimensions
    width = item.get("width")
    height = item.get("height")

    if (
        not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or width <= 0
        or height <= 0
    ):
        invalid_dimensions.append(image_id)

    # Champs obligatoires
    for field in ["file_name", "width", "height", "objects", "captions"]:
        if field not in item:
            missing_fields.append((image_id, field))

    # Vérification des objets
    for obj in item.get("objects", []):

        for field in ["category", "bbox", "area", "iscrowd"]:
            if field not in obj:
                missing_fields.append((image_id, field))

        category = obj.get("category")

        if category not in valid_categories:
            unknown_categories[category] += 1

        bbox = obj.get("bbox")

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(v, (int, float)) for v in bbox
            )
        ):
            invalid_bboxes.append(image_id)
            continue

        x, y, w, h = bbox

        if (
            x < 0
            or y < 0
            or w <= 0
            or h <= 0
            or x + w > item["width"] + 1
            or y + h > item["height"] + 1
        ):
            invalid_bboxes.append(image_id)

        area = obj.get("area")

        if (
            not isinstance(area, (int, float))
            or area <= 0
        ):
            invalid_areas.append(image_id)

# ==========================================
# 6. RAPPORT FINAL
# ==========================================

total_objects = sum(len(item["objects"]) for item in metadata)

print("\n========== METADATA QUALITY CONTROL ==========")

print("Total images:", len(metadata))
print("Total objects:", total_objects)

print("\n--- Quality checks ---")

print("Duplicate image IDs:", len(duplicate_image_ids))
print("Invalid image dimensions:", len(invalid_dimensions))
print("Missing fields:", len(missing_fields))
print("Unknown category objects:", sum(unknown_categories.values()))
print("Invalid bounding boxes:", len(invalid_bboxes))
print("Invalid object areas:", len(invalid_areas))

print("\n========== VALIDATION STATUS ==========")

if not any([
    duplicate_image_ids,
    invalid_dimensions,
    missing_fields,
    unknown_categories,
    invalid_bboxes,
    invalid_areas
]):
    print("SUCCESS: All metadata quality checks passed.")
else:
    print("WARNING: Some metadata quality issues were detected.")

print("\nMetadata quality control completed.")