from pathlib import Path
from PIL import Image
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

FOLDERS = {
    "train": DATASET_DIR / "preprocessed" / "train2017",
    "validation": DATASET_DIR / "preprocessed" / "val2017",
}

def audit_folder(name, folder):
    print("\n" + "=" * 60)
    print(f"AUDIT : {name.upper()}")
    print("=" * 60)

    total = 0
    errors = 0
    modes = Counter()
    dimensions = Counter()
    exif_orientations = Counter()

    for path in folder.iterdir():
        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                total += 1

                modes[img.mode] += 1
                dimensions[img.size] += 1

                orientation = img.getexif().get(274, 1)
                exif_orientations[orientation] += 1

                img.verify()

        except Exception as e:
            errors += 1
            print(f"Erreur : {path.name} -> {e}")

    print(f"Nombre total : {total}")
    print(f"Images illisibles : {errors}")

    print("\nModes couleur :")
    for mode, count in modes.items():
        print(f"{mode}: {count}")

    print("\nDimensions les plus fréquentes :")
    for size, count in dimensions.most_common(10):
        print(f"{size}: {count}")

    print("\nOrientations EXIF :")
    for orientation, count in exif_orientations.items():
        print(f"Orientation {orientation}: {count}")

    rgb_count = modes.get("RGB", 0)

    print("\nRésultat RGB :")
    if rgb_count == total and total > 0:
        print("OK : toutes les images sont en RGB.")
    else:
        print("ATTENTION : certaines images ne sont pas en RGB.")

for name, folder in FOLDERS.items():
    if folder.exists():
        audit_folder(name, folder)
    else:
        print(f"Dossier introuvable : {folder}")