
import json
from pathlib import Path

DATA_DIR = Path("data/processed")
SPLITS = ["knowledge_base", "validation", "test"]

def extract_metadata(split):
    captions_path = DATA_DIR / split / "captions.json"

    with open(captions_path, "r") as f:
        coco_data = json.load(f)

    # Group captions by image ID
    captions_by_image = {}

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        captions_by_image.setdefault(image_id, []).append(
            annotation["caption"]
        )

    metadata = []

    for image in coco_data["images"]:
        width = image["width"]
        height = image["height"]
        image_id = image["id"]

        record = {
            "image_id": image_id,
            "file_name": image["file_name"],
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 3),
            "captions": captions_by_image.get(image_id, []),
            "num_captions": len(captions_by_image.get(image_id, []))
        }

        metadata.append(record)

    output_dir = DATA_DIR / split
    output_path = output_dir / "metadata.json"

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"{split}: {len(metadata)} images")
    print(f"Metadata saved: {output_path}")

if __name__ == "__main__":
    for split in SPLITS:
        extract_metadata(split)
