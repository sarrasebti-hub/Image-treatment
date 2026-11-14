
import json
import random
from pathlib import Path

# Reproducibility: same selection every time
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "dataset" / "annotations"
OUTPUT_DIR = BASE_DIR / "dataset"

TRAIN_SIZE = 10_000
VALIDATION_SIZE = 100
TEST_SIZE = 1_000


def load_json(filename):
    path = ANNOTATIONS_DIR / filename

    print(f"Loading {filename}...")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # Load COCO annotation files
    train_data = load_json("captions_train2017.json")
    val_data = load_json("captions_val2017.json")

    # Get unique image IDs
    train_ids = sorted(
        image["id"] for image in train_data["images"]
    )

    test_ids = sorted(
        image["id"] for image in val_data["images"]
    )

    # Verify that enough images are available
    assert len(train_ids) >= TRAIN_SIZE + VALIDATION_SIZE
    assert len(test_ids) >= TEST_SIZE

    # Shuffle train2017 IDs deterministically
    random.shuffle(train_ids)

    # Select disjoint train and validation sets
    selected_train = sorted(train_ids[:TRAIN_SIZE])
    selected_validation = sorted(
        train_ids[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    )

    # Select test images from val2017
    selected_test = sorted(random.sample(test_ids, TEST_SIZE))

    # Verify no overlap
    assert set(selected_train).isdisjoint(selected_validation)
    assert set(selected_train).isdisjoint(selected_test)
    assert set(selected_validation).isdisjoint(selected_test)

    # Save image IDs
    selections = {
        "train": selected_train,
        "validation": selected_validation,
        "test": selected_test
    }

    output_path = OUTPUT_DIR / "dataset_splits.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(selections, file, indent=4)

    print("\nDataset selection completed!")
    print(f"Train images: {len(selected_train)}")
    print(f"Validation images: {len(selected_validation)}")
    print(f"Test images: {len(selected_test)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
