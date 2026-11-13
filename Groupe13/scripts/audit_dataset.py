
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

DATASETS = {
    "train2017": {
        "images": DATA_DIR / "train2017",
        "captions": DATA_DIR / "annotations" / "captions_train2017.json"
    },
    "val2017": {
        "images": DATA_DIR / "val2017",
        "captions": DATA_DIR / "annotations" / "captions_val2017.json"
    }
}

# ==========================================
# 2. AUDIT D'UN DATASET
# ==========================================

def audit_dataset(name, image_dir, caption_file):

    print("\n" + "=" * 55)
    print(f"AUDIT : {name}")
    print("=" * 55)

    report = {}

    if not image_dir.exists():
        print("ERREUR : dossier images introuvable")
        return {"error": "Image directory missing"}

    if not caption_file.exists():
        print("ERREUR : fichier captions introuvable")
        return {"error": "Caption file missing"}

    # Charger les annotations
    with open(caption_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_info = data.get("images", [])
    annotations = data.get("annotations", [])

    # Compter les images physiques
    image_files = list(image_dir.glob("*.jpg"))

    # IDs des images dans les annotations
    image_ids = [img["id"] for img in images_info]

    # Vérifier les IDs dupliqués
    duplicate_ids = len(image_ids) - len(set(image_ids))

    # Compter les captions par image
    captions_per_image = Counter(
        ann["image_id"] for ann in annotations
    )

    # Vérifier les cinq captions
    caption_distribution = Counter(
        captions_per_image.get(image_id, 0)
        for image_id in image_ids
    )

    incorrect_caption_count = {
        image_id: captions_per_image.get(image_id, 0)
        for image_id in image_ids
        if captions_per_image.get(image_id, 0) != 5
    }

    # Vérifier les images manquantes et corrompues
    missing_images = []
    corrupted_images = []
    hashes = defaultdict(list)

    for image_id in image_ids:

        filename = f"{image_id:012d}.jpg"
        image_path = image_dir / filename

        if not image_path.exists():
            missing_images.append(filename)
            continue

        try:
            # Vérification de l'image
            with Image.open(image_path) as img:
                img.verify()

            # Calcul du hash pour détecter les doublons exacts
            file_hash = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()

            hashes[file_hash].append(filename)

        except Exception:
            corrupted_images.append(filename)

    # Détecter les doublons exacts
    duplicate_images = [
        files for files in hashes.values()
        if len(files) > 1
    ]

    # Vérifier les captions sans image correspondante
    annotation_image_ids = set(image_ids)

    orphan_captions = [
        ann for ann in annotations
        if ann["image_id"] not in annotation_image_ids
    ]

    # ==========================================
    # 3. AFFICHAGE DES RESULTATS
    # ==========================================

    report = {
        "dataset": name,
        "physical_images": len(image_files),
        "annotation_images": len(images_info),
        "total_captions": len(annotations),
        "duplicate_ids": duplicate_ids,
        "caption_distribution": dict(caption_distribution),
        "incorrect_caption_count": len(incorrect_caption_count),
        "missing_images": len(missing_images),
        "corrupted_images": len(corrupted_images),
        "duplicate_image_groups": len(duplicate_images),
        "orphan_captions": len(orphan_captions)
    }

    for key, value in report.items():
        print(f"{key}: {value}")

    return report


# ==========================================
# 4. EXECUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":

    REPORT_DIR.mkdir(exist_ok=True)

    final_report = {}

    for name, paths in DATASETS.items():

        final_report[name] = audit_dataset(
            name,
            paths["images"],
            paths["captions"]
        )

    # Sauvegarder le rapport
    output_file = REPORT_DIR / "dataset_audit.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            final_report,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("AUDIT TERMINE")
    print(f"Rapport sauvegarde : {output_file}")
    print("=" * 55)
