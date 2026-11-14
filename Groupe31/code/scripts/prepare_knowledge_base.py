
import json
import random
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_DIR = PROJECT_ROOT / "datasets" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "splits"
OUTPUT_DIR = PROJECT_ROOT / "results" / "knowledge_base"

CAPTIONS_FILE = ANNOTATIONS_DIR / "captions_train2017.json"
INSTANCES_FILE = ANNOTATIONS_DIR / "instances_train2017.json"
KB_IDS_FILE = SPLITS_DIR / "knowledge_base_ids.json"

OUTPUT_FILE = OUTPUT_DIR / "knowledge_base.jsonl"

SEED = 42
CAPTIONS_PER_IMAGE = 5


# --------------------------------------------------
# 2. LOAD JSON FILES
# --------------------------------------------------

def load_json(file_path):
    print(f"Loading: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# 3. MAIN PREPARATION
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kb_ids = load_json(KB_IDS_FILE)
    captions_data = load_json(CAPTIONS_FILE)
    instances_data = load_json(INSTANCES_FILE)

    kb_id_set = set(kb_ids)

    # Image ID -> image information
    images = {
        image["id"]: image
        for image in captions_data["images"]
    }

    # Image ID -> list of captions
    captions_by_image = defaultdict(list)

    for annotation in captions_data["annotations"]:
        captions_by_image[annotation["image_id"]].append(
            annotation["caption"]
        )

    # Category ID -> category name
    categories = {
        category["id"]: category["name"]
        for category in instances_data["categories"]
    }

    # Image ID -> list of objects
    objects_by_image = defaultdict(list)

    for annotation in instances_data["annotations"]:

        image_id = annotation["image_id"]

        if image_id not in kb_id_set:
            continue

        category_id = annotation["category_id"]

        objects_by_image[image_id].append({
            "category_id": category_id,
            "category_name": categories[category_id],
            "bbox": annotation["bbox"]
        })

    # --------------------------------------------------
    # 4. SELECT FIVE CAPTIONS PER IMAGE
    # --------------------------------------------------

    rng = random.Random(SEED)

    records = []
    images_with_unusual_caption_count = 0

    for image_id in kb_ids:

        if image_id not in images:
            raise ValueError(f"Image ID {image_id} not found")

        image_info = images[image_id]
        captions = captions_by_image[image_id]

        if len(captions) < CAPTIONS_PER_IMAGE:
            raise ValueError(
                f"Image {image_id} has only {len(captions)} captions"
            )

        if len(captions) != CAPTIONS_PER_IMAGE:
            images_with_unusual_caption_count += 1

        selected_captions = rng.sample(
            captions,
            CAPTIONS_PER_IMAGE
        )

        record = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "width": image_info["width"],
            "height": image_info["height"],
            "captions": selected_captions,
            "objects": objects_by_image[image_id]
        }

        records.append(record)

    # --------------------------------------------------
    # 5. SAVE JSONL FILE
    # --------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # --------------------------------------------------
    # 6. SUMMARY
    # --------------------------------------------------

    total_objects = sum(
        len(record["objects"])
        for record in records
    )

    print("\n========== KNOWLEDGE BASE PREPARATION COMPLETE ==========")
    print(f"Images processed: {len(records)}")
    print(f"Captions per image: {CAPTIONS_PER_IMAGE}")
    print(f"Total captions: {len(records) * CAPTIONS_PER_IMAGE}")
    print(f"Total object annotations: {total_objects}")
    print(f"Images with unusual caption counts: {images_with_unusual_caption_count}")
    print(f"Random seed: {SEED}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
