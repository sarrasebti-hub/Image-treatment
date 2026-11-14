
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIR = BASE_DIR / "dataset" / "annotations"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"
SPLITS_FILE = BASE_DIR / "dataset" / "dataset_splits.json"


def load_json(path):
    print(f"Loading {path.name}...")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_split(name, source_data, selected_ids):
    selected_ids = set(selected_ids)

    images = [
        image for image in source_data["images"]
        if image["id"] in selected_ids
    ]

    annotations = [
        annotation for annotation in source_data["annotations"]
        if annotation["image_id"] in selected_ids
    ]

    result = {
        "split": name,
        "images": images,
        "annotations": annotations
    }

    output_path = OUTPUT_DIR / f"{name}_captions.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False)

    caption_counts = Counter(
        annotation["image_id"] for annotation in annotations
    )

    print(f"\n{name.upper()} dataset:")
    print(f"Images: {len(images)}")
    print(f"Captions: {len(annotations)}")

    if caption_counts:
        print(
            "Captions per image:",
            f"minimum = {min(caption_counts.values())},",
            f"maximum = {max(caption_counts.values())}"
        )

    print(f"Saved: {output_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_json(SPLITS_FILE)

    train_data = load_json(
        ANNOTATIONS_DIR / "captions_train2017.json"
    )

    val_data = load_json(
        ANNOTATIONS_DIR / "captions_val2017.json"
    )

    save_split("train", train_data, splits["train"])

    save_split("validation", train_data, splits["validation"])

    save_split("test", val_data, splits["test"])

    print("\nCaption preparation completed!")


if __name__ == "__main__":
    main()
