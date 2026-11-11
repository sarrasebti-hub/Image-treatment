import json
import random
import shutil
from pathlib import Path

ANNOTATIONS = Path("data/raw/annotations/annotations/captions_val2017.json")
SOURCE = Path("data/raw/images/val2017")
TEST = Path("data/processed/test")
IMAGES = TEST / "images"
METADATA = TEST / "metadata.json"

with open(ANNOTATIONS) as f:
    coco = json.load(f)

with open(METADATA) as f:
    old_metadata = json.load(f)

old_ids = {item["image_id"] for item in old_metadata}

available = [
    img for img in coco["images"]
    if img["id"] not in old_ids
]

random.Random(42).shuffle(available)
selected = available[:900]

captions_by_id = {}
for annotation in coco["annotations"]:
    captions_by_id.setdefault(annotation["image_id"], []).append(
        annotation["caption"]
    )

new_metadata = []

for img in selected:
    image_id = img["id"]
    filename = img["file_name"]

    source = SOURCE / filename
    destination = IMAGES / filename

    if not source.exists():
        raise FileNotFoundError(source)

    shutil.copy2(source, destination)

    new_metadata.append({
        "image_id": image_id,
        "file_name": filename,
        "width": img["width"],
        "height": img["height"],
        "aspect_ratio": round(img["width"] / img["height"], 3),
        "captions": captions_by_id.get(image_id, []),
        "num_captions": len(captions_by_id.get(image_id, []))
    })

# Backup old metadata
shutil.copy2(METADATA, TEST / "metadata_before_expansion.json")

# Preserve old records and append new ones
combined = old_metadata + new_metadata

with open(METADATA, "w") as f:
    json.dump(combined, f, indent=2)

print("Old images:", len(old_metadata))
print("New images:", len(new_metadata))
print("Total Test images:", len(combined))
