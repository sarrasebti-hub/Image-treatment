
import json
import re
from itertools import combinations
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = (
    PROJECT_ROOT / "results" / "knowledge_base"
    / "knowledge_base.jsonl"
)

CLEANED_FILE = (
    PROJECT_ROOT / "results" / "knowledge_base"
    / "knowledge_base_cleaned_captions.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "image_metadata.jsonl"


# Caption-based indicators
KEYWORDS = {
    "colours": [
        "red", "blue", "green", "yellow", "black",
        "white", "brown", "gray", "grey", "orange",
        "pink", "purple"
    ],
    "shapes": [
        "round", "rounded", "square", "rectangular",
        "oval", "triangular", "circular"
    ],
    "indoor": [
        "indoors", "inside", "room", "kitchen",
        "bedroom", "living room", "office"
    ],
    "outdoor": [
        "outdoors", "outside", "street", "road",
        "park", "beach", "field"
    ],
    "daytime": [
        "daytime", "during the day", "daylight"
    ],
    "nighttime": [
        "nighttime", "at night", "during the night"
    ]
}


def find_mentions(captions, terms):
    """Find terms explicitly mentioned in original captions."""
    found = set()

    for caption in captions:
        text = caption.lower()

        for term in terms:
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

            if re.search(pattern, text):
                found.add(term)

    return sorted(found)


def get_center(bbox):
    """COCO bbox format: [x, y, width, height]."""
    x, y, width, height = bbox

    return (
        x + width / 2,
        y + height / 2
    )


def calculate_overlap(bbox_a, bbox_b):
    """Intersection area divided by the smaller box area."""
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    intersection = max(0, right - left) * max(0, bottom - top)

    area_a = aw * ah
    area_b = bw * bh
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0

    return intersection / smaller_area


def derive_relationships(objects):
    """Infer one spatial relationship for each unordered object pair."""
    relationships = []

    for obj_a, obj_b in combinations(objects, 2):

        center_a = get_center(obj_a["bbox"])
        center_b = get_center(obj_b["bbox"])

        overlap = calculate_overlap(
            obj_a["bbox"],
            obj_b["bbox"]
        )

        if overlap >= 0.10:
            relation = "overlapping"

        elif abs(center_a[0] - center_b[0]) >= abs(
            center_a[1] - center_b[1]
        ):
            relation = (
                "left_of" if center_a[0] < center_b[0]
                else "right_of"
            )

        else:
            relation = (
                "above" if center_a[1] < center_b[1]
                else "below"
            )

        relationships.append({
            "object_a": obj_a["category_name"],
            "object_b": obj_b["category_name"],
            "relation": relation
        })

    return relationships


# Load cleaned captions, preserving the original versions separately
cleaned_records = {}

with open(CLEANED_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        cleaned_records[record["image_id"]] = record


total_images = 0
total_objects = 0
total_relationships = 0
failed_records = 0

print("Building image-level metadata...")

with open(KB_FILE, "r", encoding="utf-8") as input_file, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:

    for line in input_file:
        record = json.loads(line)

        try:
            image_id = record["image_id"]
            captions = record["captions"]
            objects = record["objects"]

            cleaned_record = cleaned_records[image_id]

            if cleaned_record["original_captions"] != captions:
                raise ValueError("Caption references do not match")

            relationships = derive_relationships(objects)

            metadata = {
                "image_id": image_id,
                "file_name": record["file_name"],

                "objects": objects,

                "object_categories": sorted({
                    obj["category_name"] for obj in objects
                }),

                "spatial_relationships": relationships,

                "caption_mentions": {
                    "colours": find_mentions(
                        captions, KEYWORDS["colours"]
                    ),
                    "shapes": find_mentions(
                        captions, KEYWORDS["shapes"]
                    ),
                    "indoor": find_mentions(
                        captions, KEYWORDS["indoor"]
                    ),
                    "outdoor": find_mentions(
                        captions, KEYWORDS["outdoor"]
                    ),
                    "daytime": find_mentions(
                        captions, KEYWORDS["daytime"]
                    ),
                    "nighttime": find_mentions(
                        captions, KEYWORDS["nighttime"]
                    )
                }
            }

            output_file.write(
                json.dumps(metadata, ensure_ascii=False) + "\n"
            )

            total_images += 1
            total_objects += len(objects)
            total_relationships += len(relationships)

        except Exception as error:
            failed_records += 1
            print(f"Error in image {record.get('image_id')}: {error}")


print("\n========== IMAGE METADATA REPORT ==========")
print(f"Images processed: {total_images}")
print(f"Object annotations: {total_objects}")
print(f"Spatial relationships: {total_relationships}")
print(f"Failed records: {failed_records}")
print(f"Output file: {OUTPUT_FILE}")
print("\nMetadata preparation completed.")
