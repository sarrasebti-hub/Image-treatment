
import json
import random
import sys
from pathlib import Path


# ============================================================
# 1. IMPORT CENTRALIZED CONFIGURATION
# ============================================================

# Add 01_src to Python's module search path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from config.pipeline_config import (
    TRAIN_CAPTIONS_FILE,
    TEST_CAPTIONS_FILE,
    TRAIN_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TRAIN_SOURCE,
    TEST_SOURCE,
    SPLITS_DIR,
    RANDOM_SEED,
    TRAINING_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
    TOTAL_TRAIN_POOL,
    CAPTIONS_PER_IMAGE,
    get_pipeline_snapshot,
)


# ============================================================
# 2. LOAD IMAGE IDS
# ============================================================

def load_image_ids(annotation_file):
    """Load and sort image IDs from a COCO captions file."""

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    with open(annotation_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    return sorted(
        image["id"] for image in data["images"]
    )


# ============================================================
# 3. CHECK INPUTS AND CONFIGURATION
# ============================================================

def check_inputs():
    """Verify that required inputs and configuration are valid."""

    required_paths = [
        TRAIN_CAPTIONS_FILE,
        TEST_CAPTIONS_FILE,
        TRAIN_IMAGES_DIR,
        TEST_IMAGES_DIR,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file or directory not found: {path}"
            )

    if (
        TRAINING_SIZE <= 0
        or VALIDATION_SIZE <= 0
        or TEST_SIZE <= 0
    ):
        raise ValueError(
            "All split sizes must be positive."
        )

    if CAPTIONS_PER_IMAGE <= 0:
        raise ValueError(
            "CAPTIONS_PER_IMAGE must be positive."
        )

    if TOTAL_TRAIN_POOL != TRAINING_SIZE + VALIDATION_SIZE:
        raise ValueError(
            "TOTAL_TRAIN_POOL must equal "
            "TRAINING_SIZE + VALIDATION_SIZE."
        )


# ============================================================
# 4. CREATE SPLITS
# ============================================================

def create_splits():
    """Create reproducible training, validation and test splits."""

    check_inputs()

    train_pool_ids = load_image_ids(TRAIN_CAPTIONS_FILE)
    test_pool_ids = load_image_ids(TEST_CAPTIONS_FILE)

    print(
        "Available training images:",
        len(train_pool_ids)
    )

    print(
        "Available test-pool images:",
        len(test_pool_ids)
    )

    if len(train_pool_ids) < TOTAL_TRAIN_POOL:
        raise ValueError(
            "Not enough images in the COCO training source."
        )

    if len(test_pool_ids) < TEST_SIZE:
        raise ValueError(
            "Not enough images in the COCO test source."
        )

    # Local random generator ensures reproducibility
    # without modifying Python's global random state.
    rng = random.Random(RANDOM_SEED)

    # Select the combined training and validation pool
    selected_train_pool = rng.sample(
        train_pool_ids,
        TOTAL_TRAIN_POOL
    )

    rng.shuffle(selected_train_pool)

    # Divide the selected pool
    training_ids = selected_train_pool[:TRAINING_SIZE]

    validation_ids = selected_train_pool[
        TRAINING_SIZE:TRAINING_SIZE + VALIDATION_SIZE
    ]

    # Select test images independently from COCO val2017
    test_ids = rng.sample(
        test_pool_ids,
        TEST_SIZE
    )

    return (
        sorted(training_ids),
        sorted(validation_ids),
        sorted(test_ids),
    )


# ============================================================
# 5. CHECK SPLIT CONSISTENCY
# ============================================================

def check_splits(training_ids, validation_ids, test_ids):
    """Verify split sizes, uniqueness and disjointness."""

    training_set = set(training_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    # Verify expected sizes
    assert len(training_ids) == TRAINING_SIZE
    assert len(validation_ids) == VALIDATION_SIZE
    assert len(test_ids) == TEST_SIZE

    # Verify there are no duplicate IDs
    assert len(training_set) == len(training_ids)
    assert len(validation_set) == len(validation_ids)
    assert len(test_set) == len(test_ids)

    # Verify that the splits are mutually disjoint
    assert training_set.isdisjoint(validation_set)
    assert training_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)

    print("Split consistency checks passed.")


# ============================================================
# 6. SAVE SPLITS AND CONFIGURATION
# ============================================================

def save_splits(training_ids, validation_ids, test_ids):
    """Save split IDs and their configuration metadata."""

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    split_lists = {
        "training_ids.json": training_ids,
        "validation_ids.json": validation_ids,
        "test_ids.json": test_ids,
    }

    # Save image IDs
    for filename, ids in split_lists.items():

        output_file = SPLITS_DIR / filename

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(ids, file, indent=4)

        print(f"Saved: {output_file}")

    # Build configuration metadata using centralized values
    config = {
        "pipeline_configuration": get_pipeline_snapshot(),
        "random_seed": RANDOM_SEED,
        "source_training_dataset": f"COCO {TRAIN_SOURCE}",
        "source_test_dataset": f"COCO {TEST_SOURCE}",
        "total_selected_from_train2017": TOTAL_TRAIN_POOL,

        "training_images": len(training_ids),
        "validation_images": len(validation_ids),
        "test_images": len(test_ids),

        "captions_per_image": CAPTIONS_PER_IMAGE,

        "expected_training_captions": (
            len(training_ids) * CAPTIONS_PER_IMAGE
        ),

        "expected_validation_captions": (
            len(validation_ids) * CAPTIONS_PER_IMAGE
        ),

        "expected_test_captions": (
            len(test_ids) * CAPTIONS_PER_IMAGE
        ),

        "selection_method": (
            "Random sampling with fixed seed"
        ),
    }

    config_file = SPLITS_DIR / "split_config.json"

    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Saved: {config_file}")


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():

    print("Starting dataset splitting...\n")

    # Create the splits
    training_ids, validation_ids, test_ids = create_splits()

    # Verify the splits
    check_splits(
        training_ids,
        validation_ids,
        test_ids
    )

    # Save the splits and configuration
    save_splits(
        training_ids,
        validation_ids,
        test_ids
    )

    print("\nDataset splitting completed successfully!")

    print("\nFinal split sizes:")
    print("Training:", len(training_ids))
    print("Validation:", len(validation_ids))
    print("Test:", len(test_ids))

    print(
        "Expected training captions:",
        len(training_ids) * CAPTIONS_PER_IMAGE
    )

    print(
        "Expected validation captions:",
        len(validation_ids) * CAPTIONS_PER_IMAGE
    )

    print(
        "Expected test captions:",
        len(test_ids) * CAPTIONS_PER_IMAGE
    )


if __name__ == "__main__":
    main()
