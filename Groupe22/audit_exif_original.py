from pathlib import Path
from PIL import Image
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

FOLDERS = {
    "TRAIN": DATASET_DIR / "train2017",
    "VALIDATION": DATASET_DIR / "val2017",
}

for name, folder in FOLDERS.items():

    print("\n" + "=" * 55)
    print(f"ORIENTATION EXIF DES IMAGES ORIGINALES : {name}")
    print("=" * 55)

    orientations = Counter()
    errors = 0
    total = 0

    for path in folder.iterdir():

        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                total += 1

                orientation = img.getexif().get(274, 1)
                orientations[orientation] += 1

        except Exception as e:
            errors += 1
            print(f"Erreur : {path.name} -> {e}")

    print(f"Total : {total}")
    print(f"Erreurs : {errors}")

    print("\nRépartition des orientations :")

    for orientation, count in sorted(orientations.items()):
        print(f"Orientation {orientation}: {count}")

    needs_correction = sum(
        count for orientation, count in orientations.items()
        if orientation != 1
    )

    print(f"\nImages avec orientation à corriger : {needs_correction}")