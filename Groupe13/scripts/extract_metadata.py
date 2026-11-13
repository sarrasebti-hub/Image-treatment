
from pathlib import Path
import json
import csv
from collections import Counter, defaultdict
from itertools import combinations


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIR = ROOT / "data" / "annotations"
SPLITS_DIR = ROOT / "data" / "splits"
CAPTIONS_DIR = ROOT / "data" / "captions"
OUTPUT_DIR = ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INSTANCES_FILE = ANNOTATIONS_DIR / "instances_train2017.json"
KB_IDS_FILE = SPLITS_DIR / "knowledge_base_image_ids.json"
CAPTIONS_FILE = CAPTIONS_DIR / "kb_captions_raw.json"


# ==========================================================
# 2. CHARGEMENT DES DONNEES
# ==========================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================================
# 3. DETECTION DES RELATIONS SPATIALES
# ==========================================================

def get_spatial_relation(box_a, box_b):

    # COCO bbox format: [x, y, width, height]
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    center_ax = ax + aw / 2
    center_ay = ay + ah / 2

    center_bx = bx + bw / 2
    center_by = by + bh / 2

    relations = []

    if center_ax < center_bx:
        relations.append("left_of")

    elif center_ax > center_bx:
        relations.append("right_of")

    if center_ay < center_by:
        relations.append("above")

    elif center_ay > center_by:
        relations.append("below")

    return relations


# ==========================================================
# 4. EXTRACTION DES METADONNEES
# ==========================================================

def extract_metadata(instances, kb_ids, captions):

    kb_set = set(kb_ids)

    categories = {
        category["id"]: category["name"]
        for category in instances["categories"]
    }

    images_info = {
        image["id"]: image
        for image in instances["images"]
        if image["id"] in kb_set
    }

    objects_by_image = defaultdict(list)

    for annotation in instances["annotations"]:

        image_id = annotation["image_id"]

        if image_id not in kb_set:
            continue

        category_id = annotation["category_id"]

        obj = {
            "annotation_id": annotation["id"],
            "category_id": category_id,
            "category": categories.get(category_id, "unknown"),
            "bbox": annotation.get("bbox"),
            "area": annotation.get("area"),
            "iscrowd": annotation.get("iscrowd", 0)
        }

        objects_by_image[image_id].append(obj)

    captions_by_image = defaultdict(list)

    for item in captions:
        captions_by_image[item["image_id"]].append(
            item["caption"]
        )

    metadata = []

    for image_id in kb_ids:

        image_info = images_info.get(image_id, {})

        objects = objects_by_image.get(image_id, [])

        spatial_relations = []

        # Calcul des relations entre paires d'objets
        for obj_a, obj_b in combinations(objects, 2):

            if not obj_a["bbox"] or not obj_b["bbox"]:
                continue

            relations = get_spatial_relation(
                obj_a["bbox"],
                obj_b["bbox"]
            )

            for relation in relations:

                spatial_relations.append({
                    "object_1": obj_a["category"],
                    "object_2": obj_b["category"],
                    "relation": relation
                })

        metadata.append({
            "image_id": image_id,
            "file_name": image_info.get("file_name"),
            "width": image_info.get("width"),
            "height": image_info.get("height"),
            "captions": captions_by_image.get(image_id, []),
            "objects": objects,
            "object_count": len(objects),
            "spatial_relations": spatial_relations
        })

    return metadata


# ==========================================================
# 5. STATISTIQUES DES METADONNEES
# ==========================================================

def calculate_statistics(metadata):

    category_counter = Counter()
    relation_counter = Counter()

    total_objects = 0
    total_relations = 0

    for image in metadata:

        total_objects += image["object_count"]

        for obj in image["objects"]:
            category_counter[obj["category"]] += 1

        for relation in image["spatial_relations"]:
            relation_counter[relation["relation"]] += 1

            total_relations += 1

    statistics = {
        "total_images": len(metadata),
        "total_objects": total_objects,
        "average_objects_per_image": round(
            total_objects / len(metadata), 2
        ) if metadata else 0,
        "total_spatial_relations": total_relations,
        "object_categories": dict(category_counter),
        "spatial_relation_types": dict(relation_counter)
    }

    return statistics


# ==========================================================
# 6. SAUVEGARDE JSON ET CSV
# ==========================================================

def save_results(metadata, statistics):

    json_path = OUTPUT_DIR / "kb_metadata.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    csv_path = OUTPUT_DIR / "kb_metadata_summary.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_id",
            "file_name",
            "width",
            "height",
            "object_count",
            "caption_count",
            "spatial_relation_count"
        ])

        for image in metadata:

            writer.writerow([
                image["image_id"],
                image["file_name"],
                image["width"],
                image["height"],
                image["object_count"],
                len(image["captions"]),
                len(image["spatial_relations"])
            ])

    stats_path = OUTPUT_DIR / "metadata_statistics.json"

    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(
            statistics,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {stats_path}")


# ==========================================================
# 7. MAIN
# ==========================================================

def main():

    print("=" * 55)
    print("COCO METADATA EXTRACTION")
    print("=" * 55)

    instances = load_json(INSTANCES_FILE)
    kb_ids = load_json(KB_IDS_FILE)
    captions = load_json(CAPTIONS_FILE)

    print(f"Knowledge Base images: {len(kb_ids)}")

    metadata = extract_metadata(
        instances,
        kb_ids,
        captions
    )

    statistics = calculate_statistics(metadata)

    save_results(metadata, statistics)

    print("\n========== METADATA STATISTICS ==========")

    print(f"Images: {statistics['total_images']}")
    print(f"Objects: {statistics['total_objects']}")
    print(
        "Average objects per image:",
        statistics["average_objects_per_image"]
    )
    print(
        "Spatial relations:",
        statistics["total_spatial_relations"]
    )

    print("\nMETADATA EXTRACTION COMPLETED")


if __name__ == "__main__":
    main()
