
import json
import hashlib
from collections import Counter
from pathlib import Path

import sys


# Add 01_src to Python's module search path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from config.pipeline_config import (  # noqa: E402
    CAPTIONS_PER_IMAGE,
    PROCESSED_DATA,
    SPLITS_DIR,
    TEST_SIZE,
    TRAINING_SIZE,
    VALIDATION_SIZE,
    validate_saved_pipeline_compatibility,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

EXPECTED_SIZES = {
    "training": TRAINING_SIZE,
    "validation": VALIDATION_SIZE,
    "test": TEST_SIZE,
}

EXPECTED_CAPTIONS_PER_IMAGE = CAPTIONS_PER_IMAGE


# ============================================================
# 2. GENERAL HELPER FUNCTIONS
# ============================================================

def load_json(file_path):
    """Load a JSON file."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_file_hash(file_path):
    """
    Calculate the SHA-256 hash of a file.

    Reading in chunks avoids loading the entire image
    into memory.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 3. LOAD SPLIT IDS
# ============================================================

def load_split_ids():

    split_ids = {}

    for split_name in EXPECTED_SIZES:

        file_path = SPLITS_DIR / f"{split_name}_ids.json"

        ids = load_json(file_path)

        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate IDs found in {split_name} ID list."
            )

        split_ids[split_name] = set(ids)

    return split_ids


# ============================================================
# 4. CHECK SPLIT SIZES AND ID OVERLAP
# ============================================================

def validate_split_integrity(split_ids):

    print("\n[1] SPLIT INTEGRITY")

    for split_name, expected_size in EXPECTED_SIZES.items():

        actual_size = len(split_ids[split_name])

        if actual_size != expected_size:
            raise ValueError(
                f"{split_name}: expected {expected_size} IDs, "
                f"found {actual_size}."
            )

        print(f"  {split_name}: {actual_size} IDs - OK")

    # Check intersections between splits
    split_names = list(split_ids.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):

            first = split_names[i]
            second = split_names[j]

            overlap = split_ids[first] & split_ids[second]

            if overlap:
                raise ValueError(
                    f"ID leakage between {first} and {second}: "
                    f"{len(overlap)} overlapping IDs."
                )

    print("  No image ID overlap between splits - OK")


# ============================================================
# 5. VALIDATE ONE PROCESSED SPLIT
# ============================================================

def validate_one_split(split_name, expected_ids):

    print(f"\nChecking {split_name}...")

    split_dir = PROCESSED_DATA / split_name

    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"

    captions_file = annotations_dir / "captions.json"
    instances_file = annotations_dir / "instances.json"

    captions_data = load_json(captions_file)
    instances_data = load_json(instances_file)

    # --------------------------------------------------------
    # A. Check image records in both annotation files
    # --------------------------------------------------------

    caption_image_ids = {
        image["id"] for image in captions_data["images"]
    }

    instance_image_ids = {
        image["id"] for image in instances_data["images"]
    }

    if caption_image_ids != expected_ids:
        raise ValueError(
            f"{split_name}: captions image IDs do not match "
            "the saved split IDs."
        )

    if instance_image_ids != expected_ids:
        raise ValueError(
            f"{split_name}: instances image IDs do not match "
            "the saved split IDs."
        )

    if caption_image_ids != instance_image_ids:
        raise ValueError(
            f"{split_name}: captions and instances IDs differ."
        )

    print("  Annotation image IDs - OK")

    # --------------------------------------------------------
    # B. Check copied image files
    # --------------------------------------------------------

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}"
        )

    image_records = captions_data["images"]

    expected_filenames = {
        image["file_name"] for image in image_records
    }

    actual_filenames = {
        path.name for path in images_dir.iterdir()
        if path.is_file()
    }

    missing_files = expected_filenames - actual_filenames
    unexpected_files = actual_filenames - expected_filenames

    if missing_files:
        raise ValueError(
            f"{split_name}: {len(missing_files)} image files missing."
        )

    if unexpected_files:
        raise ValueError(
            f"{split_name}: {len(unexpected_files)} unexpected files."
        )

    print(f"  Image files: {len(actual_filenames)} - OK")

    # --------------------------------------------------------
    # C. Check that every image has exactly five captions
    # --------------------------------------------------------

    caption_counts = Counter(
        annotation["image_id"]
        for annotation in captions_data["annotations"]
    )

    incorrect_counts = {
        image_id: caption_counts[image_id]
        for image_id in expected_ids
        if caption_counts[image_id] != EXPECTED_CAPTIONS_PER_IMAGE
    }

    if incorrect_counts:
        raise ValueError(
            f"{split_name}: {len(incorrect_counts)} images "
            "do not have exactly five captions."
        )

    expected_caption_total = (
        len(expected_ids) * EXPECTED_CAPTIONS_PER_IMAGE
    )

    if len(captions_data["annotations"]) != expected_caption_total:
        raise ValueError(
            f"{split_name}: incorrect total caption count."
        )

    print(f"  Captions: {expected_caption_total} - OK")
    print("  Five captions per image - OK")

    # --------------------------------------------------------
    # D. Check annotation references
    # --------------------------------------------------------

    for annotation in captions_data["annotations"]:
        if annotation["image_id"] not in expected_ids:
            raise ValueError(
                f"{split_name}: caption references an unknown image."
            )

    for annotation in instances_data["annotations"]:
        if annotation["image_id"] not in expected_ids:
            raise ValueError(
                f"{split_name}: object annotation references "
                "an unknown image."
            )

    # --------------------------------------------------------
    # E. Check object category references
    # --------------------------------------------------------

    category_ids = {
        category["id"]
        for category in instances_data["categories"]
    }

    invalid_categories = [
        annotation["category_id"]
        for annotation in instances_data["annotations"]
        if annotation["category_id"] not in category_ids
    ]

    if invalid_categories:
        raise ValueError(
            f"{split_name}: invalid category references found."
        )

    print(
        f"  Object annotations: "
        f"{len(instances_data['annotations'])} - OK"
    )

    print("  Object category references - OK")


# ============================================================
# 6. CHECK REFERENCE CAPTION LEAKAGE
# ============================================================

def validate_caption_leakage():

    print("\n[2] REFERENCE CAPTION LEAKAGE")

    training_file = (
        PROCESSED_DATA
        / "training"
        / "annotations"
        / "captions.json"
    )

    validation_file = (
        PROCESSED_DATA
        / "validation"
        / "annotations"
        / "captions.json"
    )

    test_file = (
        PROCESSED_DATA
        / "test"
        / "annotations"
        / "captions.json"
    )

    training_data = load_json(training_file)
    validation_data = load_json(validation_file)
    test_data = load_json(test_file)

    training_ids = {
        image["id"] for image in training_data["images"]
    }

    # Ensure validation and test image references
    # are not present in the training annotations.
    for name, data in [
        ("validation", validation_data),
        ("test", test_data)
    ]:

        reference_ids = {
            image["id"] for image in data["images"]
        }

        overlap = training_ids & reference_ids

        if overlap:
            raise ValueError(
                f"{name} image leakage into training."
            )

    print("  Validation reference images excluded from training - OK")
    print("  Test reference images excluded from training - OK")

    # Check that all retained caption records belong
    # only to their respective split.
    training_caption_ids = {
        annotation["image_id"]
        for annotation in training_data["annotations"]
    }

    validation_caption_ids = {
        annotation["image_id"]
        for annotation in validation_data["annotations"]
    }

    test_caption_ids = {
        annotation["image_id"]
        for annotation in test_data["annotations"]
    }

    if training_caption_ids & validation_caption_ids:
        raise ValueError("Validation captions overlap with training.")

    if training_caption_ids & test_caption_ids:
        raise ValueError("Test captions overlap with training.")

    print("  Reference caption image-ID separation - OK")


# ============================================================
# 7. CHECK EXACT DUPLICATE IMAGES BETWEEN SPLITS
# ============================================================

def validate_duplicate_images():

    print("\n[3] EXACT IMAGE DUPLICATE CHECK")

    hashes_by_split = {}

    for split_name in EXPECTED_SIZES:

        images_dir = PROCESSED_DATA / split_name / "images"

        hashes = {}

        for image_file in images_dir.iterdir():

            if not image_file.is_file():
                continue

            file_hash = get_file_hash(image_file)

            hashes.setdefault(file_hash, []).append(
                image_file.name
            )

        hashes_by_split[split_name] = hashes

    split_names = list(hashes_by_split.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):

            first = split_names[i]
            second = split_names[j]

            common_hashes = (
                set(hashes_by_split[first])
                & set(hashes_by_split[second])
            )

            if common_hashes:
                raise ValueError(
                    f"Exact duplicate image content found "
                    f"between {first} and {second}: "
                    f"{len(common_hashes)} duplicate(s)."
                )

    print("  No exact duplicate image content across splits - OK")


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    validate_saved_pipeline_compatibility()
    print("Pipeline configuration compatibility - OK")

    # Load saved IDs
    split_ids = load_split_ids()

    # Check split sizes and overlaps
    validate_split_integrity(split_ids)

    # Validate each processed split
    print("\n[1] DATASET CONTENT")

    for split_name in EXPECTED_SIZES:

        validate_one_split(
            split_name,
            split_ids[split_name]
        )

    # Check reference caption separation
    validate_caption_leakage()

    # Check exact duplicate images
    validate_duplicate_images()

    print("\n" + "=" * 60)
    print("ALL DATASET VALIDATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
