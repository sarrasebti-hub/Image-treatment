
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

# ============================================================
# 1. IMPORT CENTRALIZED CONFIGURATION
# ============================================================

# Add 01_src to Python's module search path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from config.pipeline_config import (
    PROCESSED_DATA,
    SPLITS_DIR,
    TRAIN_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TRAIN_CAPTIONS_FILE,
    TEST_CAPTIONS_FILE,
    TRAIN_INSTANCES_FILE,
    TEST_INSTANCES_FILE,
    CAPTIONS_PER_IMAGE,
    TRAINING_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
)


# ============================================================
# 2. LOAD SAVED IMAGE IDS
# ============================================================

def load_split_ids(filename):
    """Load image IDs from one of the split JSON files."""

    file_path = SPLITS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        image_ids = json.load(file)

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(
            f"Duplicate image IDs found in {filename}"
        )

    return image_ids


# ============================================================
# 3. LOAD ORIGINAL COCO ANNOTATIONS
# ============================================================

def load_json(file_path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 4. FILTER OBJECT ANNOTATIONS
# ============================================================

def filter_annotations(data, selected_ids):
    """
    Keep only image records and object annotations
    belonging to the selected image IDs.
    """

    selected_set = set(selected_ids)

    filtered_images = [
        image
        for image in data["images"]
        if image["id"] in selected_set
    ]

    filtered_annotations = [
        annotation
        for annotation in data["annotations"]
        if annotation["image_id"] in selected_set
    ]

    # Preserve the original metadata
    filtered_data = data.copy()
    filtered_data["images"] = filtered_images
    filtered_data["annotations"] = filtered_annotations

    # Verify that all selected images were found
    found_ids = {image["id"] for image in filtered_images}
    missing_ids = selected_set - found_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} selected image IDs "
            "were not found in the instances file."
        )

    return filtered_data


# ============================================================
# 5. FILTER CAPTION ANNOTATIONS
# ============================================================

def filter_caption_annotations(
    data,
    selected_ids,
    captions_per_image=CAPTIONS_PER_IMAGE
):
    """
    Keep a fixed number of captions per selected image.

    Captions are selected deterministically using the
    lowest annotation IDs.

    Raises an error if an image has fewer captions
    than required.
    """

    selected_set = set(selected_ids)

    filtered_images = [
        image
        for image in data["images"]
        if image["id"] in selected_set
    ]

    found_ids = {image["id"] for image in filtered_images}
    missing_ids = selected_set - found_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} selected images "
            "were not found in the captions file."
        )

    # Group captions by image ID
    captions_by_image = {}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]

        if image_id in selected_set:
            captions_by_image.setdefault(image_id, []).append(
                annotation
            )

    retained_annotations = []
    discarded_annotations = []

    # Process images in a deterministic order
    for image_id in sorted(selected_set):

        captions = captions_by_image.get(image_id, [])

        captions = sorted(
            captions,
            key=lambda item: item["id"]
        )

        if len(captions) < captions_per_image:
            raise ValueError(
                f"Image {image_id} has only {len(captions)} "
                f"captions. Required: {captions_per_image}."
            )

        retained_annotations.extend(
            captions[:captions_per_image]
        )

        discarded_annotations.extend(
            captions[captions_per_image:]
        )

    filtered_data = data.copy()
    filtered_data["images"] = filtered_images
    filtered_data["annotations"] = retained_annotations

    return filtered_data, discarded_annotations


# ============================================================
# 6. SAVE EXTRACTED ANNOTATIONS
# ============================================================

def save_json(data, output_path):
    """Save a dictionary as a formatted JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ============================================================
# 7. COPY SELECTED IMAGES
# ============================================================

def copy_selected_images(
    image_records,
    source_dir,
    destination_dir
):
    """Copy selected original images into the processed dataset."""

    destination_dir.mkdir(parents=True, exist_ok=True)

    # Check all source images before copying
    missing_files = []

    for image in image_records:
        source_file = source_dir / image["file_name"]

        if not source_file.is_file():
            missing_files.append(source_file)

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} image files are missing. "
            f"First missing file: {missing_files[0]}"
        )

    # Copy images
    for image in image_records:
        source_file = source_dir / image["file_name"]
        destination_file = destination_dir / image["file_name"]

        shutil.copy2(source_file, destination_file)

    print(
        f"Copied {len(image_records)} images "
        f"to {destination_dir}"
    )


# ============================================================
# 8. PROCESS ONE DATASET SPLIT
# ============================================================

def process_split(
    split_name,
    selected_ids,
    captions_file,
    instances_file,
    source_images_dir,
    output_root=PROCESSED_DATA
):
    """Extract annotations and copy images for one split."""

    print(f"\nProcessing {split_name}...")

    # Load original COCO JSON files
    captions_data = load_json(captions_file)
    instances_data = load_json(instances_file)

    selected_set = set(selected_ids)

    # Check that selected IDs exist in both files
    captions_ids = {
        image["id"] for image in captions_data["images"]
    }

    instances_ids = {
        image["id"] for image in instances_data["images"]
    }

    if not selected_set.issubset(captions_ids):
        raise ValueError(
            f"Some {split_name} IDs are missing from captions."
        )

    if not selected_set.issubset(instances_ids):
        raise ValueError(
            f"Some {split_name} IDs are missing from instances."
        )

    # Extract selected captions
    filtered_captions, discarded_captions = (
        filter_caption_annotations(
            captions_data,
            selected_ids,
            captions_per_image=CAPTIONS_PER_IMAGE
        )
    )

    # Extract selected object annotations
    filtered_instances = filter_annotations(
        instances_data,
        selected_ids
    )

    # Verify that both annotation files contain the same images
    captions_image_ids = {
        image["id"] for image in filtered_captions["images"]
    }

    instances_image_ids = {
        image["id"] for image in filtered_instances["images"]
    }

    if captions_image_ids != instances_image_ids:
        raise ValueError(
            f"Image ID mismatch in {split_name} annotations."
        )

    # Create output directories
    split_dir = output_root / split_name

    images_output_dir = split_dir / "images"
    annotations_output_dir = split_dir / "annotations"

    annotations_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save captions
    save_json(
        filtered_captions,
        annotations_output_dir / "captions.json"
    )

    # Save object annotations
    save_json(
        filtered_instances,
        annotations_output_dir / "instances.json"
    )

    # Save discarded captions and selection information
    save_json(
        {
            "split": split_name,
            "selection_rule": (
                f"{CAPTIONS_PER_IMAGE} captions with the "
                "lowest annotation IDs per image"
            ),
            "discarded_caption_count": len(discarded_captions),
            "discarded_annotations": discarded_captions
        },
        annotations_output_dir / "discarded_captions.json"
    )

    # Copy selected images
    copy_selected_images(
        filtered_captions["images"],
        source_images_dir,
        images_output_dir
    )

    # Count captions per image
    caption_counts = Counter(
        annotation["image_id"]
        for annotation in filtered_captions["annotations"]
    )

    unusual_caption_counts = {
        image_id: caption_counts[image_id]
        for image_id in selected_ids
        if caption_counts[image_id] != CAPTIONS_PER_IMAGE
    }

    # Display summary
    print(f"\n{split_name} summary:")
    print(f"  Images: {len(filtered_captions['images'])}")
    print(f"  Captions: {len(filtered_captions['annotations'])}")
    print(f"  Object annotations: {len(filtered_instances['annotations'])}")
    print(f"  Discarded captions: {len(discarded_captions)}")

    if unusual_caption_counts:
        print(
            f"  WARNING: {len(unusual_caption_counts)} images "
            f"do not have exactly {CAPTIONS_PER_IMAGE} captions."
        )
    else:
        print(
            f"  Caption count check: "
            f"{CAPTIONS_PER_IMAGE} captions per image."
        )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("Starting annotation extraction...\n")

    # Load saved split IDs
    training_ids = load_split_ids("training_ids.json")
    validation_ids = load_split_ids("validation_ids.json")
    test_ids = load_split_ids("test_ids.json")

    # Verify split sizes against centralized configuration
    expected_sizes = {
        "training": TRAINING_SIZE,
        "validation": VALIDATION_SIZE,
        "test": TEST_SIZE,
    }

    actual_sizes = {
        "training": len(training_ids),
        "validation": len(validation_ids),
        "test": len(test_ids),
    }

    if actual_sizes != expected_sizes:
        raise ValueError(
            f"Split size mismatch.\n"
            f"Expected: {expected_sizes}\n"
            f"Actual: {actual_sizes}"
        )

    # Training: COCO train2017
    process_split(
        split_name="training",
        selected_ids=training_ids,
        captions_file=TRAIN_CAPTIONS_FILE,
        instances_file=TRAIN_INSTANCES_FILE,
        source_images_dir=TRAIN_IMAGES_DIR,
    )

    # Validation: COCO train2017
    process_split(
        split_name="validation",
        selected_ids=validation_ids,
        captions_file=TRAIN_CAPTIONS_FILE,
        instances_file=TRAIN_INSTANCES_FILE,
        source_images_dir=TRAIN_IMAGES_DIR,
    )

    # Test: COCO val2017
    process_split(
        split_name="test",
        selected_ids=test_ids,
        captions_file=TEST_CAPTIONS_FILE,
        instances_file=TEST_INSTANCES_FILE,
        source_images_dir=TEST_IMAGES_DIR,
    )

    print("\nAnnotation extraction completed successfully!")


if __name__ == "__main__":
    main()
