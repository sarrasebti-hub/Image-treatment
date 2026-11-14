
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

with open(DATASET_DIR / "dataset_splits.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

checks = [
    ("Train", splits["train"], DATASET_DIR / "images_train"),
    ("Validation", splits["validation"], DATASET_DIR / "images_train"),
    ("Test", splits["test"], DATASET_DIR / "images_val"),
]

for name, image_ids, folder in checks:
    valid = 0
    invalid = []

    for image_id in tqdm(image_ids, desc=f"Checking {name}"):
        path = folder / f"{image_id:012d}.jpg"

        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except Exception:
            invalid.append(image_id)

    print(f"\n{name}:")
    print(f"Expected: {len(image_ids)}")
    print(f"Valid: {valid}")
    print(f"Invalid: {len(invalid)}")

    if invalid:
        print("Invalid image IDs:", invalid[:20])
