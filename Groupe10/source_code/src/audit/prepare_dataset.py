
import json
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path("data")

RAW_IMAGES = BASE_DIR / "raw" / "images"
RAW_ANNOTATIONS = BASE_DIR / "raw" / "annotations" / "annotations"

PROCESSED_DIR = BASE_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

SPLITS = {
    "knowledge_base": {
        "ids_file": SPLITS_DIR / "knowledge_base_ids.json",
        "image_source": RAW_IMAGES / "train2017",
        "captions_source": RAW_ANNOTATIONS / "captions_train2017.json",
    },
    "validation": {
        "ids_file": SPLITS_DIR / "validation_ids.json",
        "image_source": RAW_IMAGES / "train2017",
        "captions_source": RAW_ANNOTATIONS / "captions_train2017.json",
    },
    "test": {
        "ids_file": SPLITS_DIR / "test_ids.json",
        "image_source": RAW_IMAGES / "val2017",
        "captions_source": RAW_ANNOTATIONS / "captions_val2017.json",
    },
}

# --------------------------------------------------
# 2. Load JSON
# --------------------------------------------------

def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


# --------------------------------------------------
# 3. Prepare each split
# --------------------------------------------------

def prepare_split(split_name, config):

    print(f"\nPreparing {split_name}...")

    selected_ids = set(load_json(config["ids_file"]))

    source_images = config["image_source"]
    captions_data = load_json(config["captions_source"])

    # Keep image metadata for selected IDs
    selected_images = [
        image for image in captions_data["images"]
        if image["id"] in selected_ids
    ]

    assert len(selected_images) == len(selected_ids), (
        f"Some selected IDs are missing in {split_name}"
    )

    # Keep all captions for selected images
    selected_captions = [
        annotation
        for annotation in captions_data["annotations"]
        if annotation["image_id"] in selected_ids
    ]

    # Create output folders
    output_dir = PROCESSED_DIR / split_name
    output_images = output_dir / "images"

    output_images.mkdir(parents=True, exist_ok=True)

    # Copy images and verify them
    copied = 0

    for image in selected_images:

        filename = image["file_name"]

        source = source_images / filename
        destination = output_images / filename

        if not source.exists():
            raise FileNotFoundError(source)

        shutil.copy2(source, destination)

        # Verify copied image
        with Image.open(destination) as img:
            img.verify()

        copied += 1

        if copied % 500 == 0:
            print(f"Copied {copied}/{len(selected_images)} images")

    # Save selected annotations
    output_data = {
        "images": selected_images,
        "annotations": selected_captions,
        "info": captions_data.get("info", {}),
        "licenses": captions_data.get("licenses", []),
    }

    captions_output = output_dir / "captions.json"

    with open(captions_output, "w") as file:
        json.dump(output_data, file, indent=2)

    # Verify counts
    caption_counts = Counter(
        annotation["image_id"]
        for annotation in selected_captions
    )

    assert len(selected_images) == len(selected_ids)
    assert len(list(output_images.iterdir())) == len(selected_ids)
    assert all(image_id in caption_counts for image_id in selected_ids)

    print(f"{split_name} completed:")
    print("Images:", len(selected_images))
    print("Captions:", len(selected_captions))
    print("Images with more than 5 captions:",
          sum(count > 5 for count in caption_counts.values()))


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main():

    for split_name, config in SPLITS.items():
        prepare_split(split_name, config)

    print("\n========== DATASET PREPARATION COMPLETE ==========")
    print("All three splits have been copied and verified.")
    print("Original raw dataset has NOT been modified.")


if __name__ == "__main__":
    main()
