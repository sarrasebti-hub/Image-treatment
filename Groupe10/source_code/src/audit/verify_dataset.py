
import json
from pathlib import Path
from collections import Counter
from PIL import Image

BASE_DIR = Path("data/processed")

SPLITS = ["knowledge_base", "validation", "test"]

expected_counts = {
    "knowledge_base": 10000,
    "validation": 100,
    "test": 1000
}

all_split_ids = {}

print("\n========== FINAL DATASET VERIFICATION ==========")

for split_name in SPLITS:

    print(f"\nChecking {split_name}...")

    split_dir = BASE_DIR / split_name
    images_dir = split_dir / "images"
    captions_file = split_dir / "captions.json"

    # Load annotations
    with open(captions_file, "r") as file:
        data = json.load(file)

    images = data["images"]
    annotations = data["annotations"]

    # Get image IDs
    image_ids = {image["id"] for image in images}
    all_split_ids[split_name] = image_ids

    # Check image count
    actual_files = {
        path.name for path in images_dir.iterdir()
        if path.is_file()
    }

    expected_files = {
        image["file_name"] for image in images
    }

    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files

    # Check image integrity
    corrupted_images = []

    for filename in expected_files:
        image_path = images_dir / filename

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupted_images.append(filename)

    # Check captions
    caption_ids = {
        annotation["image_id"] for annotation in annotations
    }

    invalid_caption_ids = caption_ids - image_ids
    images_without_captions = image_ids - caption_ids

    # Check counts
    caption_distribution = Counter(
        annotation["image_id"] for annotation in annotations
    )

    print("Expected images:", expected_counts[split_name])
    print("JSON images:", len(images))
    print("Image files:", len(actual_files))
    print("Captions:", len(annotations))
    print("Missing files:", len(missing_files))
    print("Extra files:", len(extra_files))
    print("Corrupted images:", len(corrupted_images))
    print("Invalid caption references:", len(invalid_caption_ids))
    print("Images without captions:", len(images_without_captions))

    # Assertions
    assert len(images) == expected_counts[split_name]
    assert len(actual_files) == expected_counts[split_name]
    assert not missing_files
    assert not extra_files
    assert not corrupted_images
    assert not invalid_caption_ids
    assert not images_without_captions

# Check overlap between splits
print("\n========== OVERLAP VERIFICATION ==========")

for i, first in enumerate(SPLITS):
    for second in SPLITS[i + 1:]:

        overlap = all_split_ids[first] & all_split_ids[second]

        print(f"{first} vs {second}: {len(overlap)} overlapping IDs")

        assert len(overlap) == 0

print("\nSUCCESS: All verification checks passed!")
