
import json
from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

SPLITS = ["knowledge_base", "validation", "test"]
DATA_DIR = Path("data/processed")

def preprocess_split(split):
    images_dir = DATA_DIR / split / "images"
    captions_file = DATA_DIR / split / "captions.json"

    with open(captions_file, "r") as f:
        data = json.load(f)

    valid_images = 0
    invalid_images = []

    for image_info in data["images"]:
        image_path = images_dir / image_info["file_name"]

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size

            valid_images += 1

        except Exception:
            invalid_images.append(image_info["file_name"])

    print(f"\n{split}:")
    print(f"Valid images: {valid_images}")
    print(f"Invalid images: {len(invalid_images)}")

    return {
        "split": split,
        "valid_images": valid_images,
        "invalid_images": invalid_images
    }

if __name__ == "__main__":
    results = []

    for split in SPLITS:
        results.append(preprocess_split(split))

    output = Path("reports/preprocessing_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nReport saved to: {output}")
