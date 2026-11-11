
import hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("data/processed")

SPLITS = [
    "knowledge_base",
    "validation",
    "test"
]

hashes = defaultdict(list)

print("Checking duplicates across selected splits...\n")

for split in SPLITS:

    image_dir = BASE_DIR / split / "images"

    for image_path in image_dir.iterdir():

        if not image_path.is_file():
            continue

        with open(image_path, "rb") as file:
            image_hash = hashlib.md5(file.read()).hexdigest()

        hashes[image_hash].append(
            (split, image_path.name)
        )

# Find exact duplicates
duplicates = {
    image_hash: files
    for image_hash, files in hashes.items()
    if len(files) > 1
}

print("Total selected images:", sum(len(files) for files in hashes.values()))
print("Duplicate groups:", len(duplicates))

for image_hash, files in duplicates.items():
    print("\nDuplicate group:")

    for split, filename in files:
        print(f"  {split}: {filename}")

print("\nCheck completed.")
