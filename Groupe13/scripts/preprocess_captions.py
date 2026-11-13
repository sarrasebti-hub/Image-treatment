
from pathlib import Path
import json
import re
from collections import Counter


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SPLITS_DIR = DATA_DIR / "splits"
CAPTIONS_DIR = DATA_DIR / "captions"

CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

CAPTIONS_FILE = ANNOTATIONS_DIR / "captions_train2017.json"

KB_IDS_FILE = SPLITS_DIR / "knowledge_base_image_ids.json"

RAW_OUTPUT = CAPTIONS_DIR / "kb_captions_raw.json"
CLEAN_OUTPUT = CAPTIONS_DIR / "kb_captions_cleaned.json"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CHARGEMENT DES FICHIERS
# ==========================================================

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================================
# 3. NETTOYAGE DES CAPTIONS
# ==========================================================

def clean_caption(caption):

    # Conversion en minuscules
    caption = caption.lower()

    # Supprimer les espaces au début et à la fin
    caption = caption.strip()

    # Normaliser les espaces multiples
    caption = re.sub(r"\s+", " ", caption)

    # Retirer les espaces avant la ponctuation
    caption = re.sub(r"\s+([.,!?;:])", r"\1", caption)

    return caption


# ==========================================================
# 4. SELECTION DES CAPTIONS DE LA KNOWLEDGE BASE
# ==========================================================

def select_kb_captions(captions_data, kb_ids):

    kb_id_set = set(kb_ids)

    raw_captions = []
    cleaned_captions = []

    for annotation in captions_data["annotations"]:

        image_id = annotation["image_id"]

        # Garder uniquement les images de la KB
        if image_id not in kb_id_set:
            continue

        original_caption = annotation["caption"]

        cleaned_caption = clean_caption(original_caption)

        # Version originale
        raw_captions.append({
            "annotation_id": annotation["id"],
            "image_id": image_id,
            "caption": original_caption
        })

        # Version nettoyée
        cleaned_captions.append({
            "annotation_id": annotation["id"],
            "image_id": image_id,
            "caption": cleaned_caption
        })

    return raw_captions, cleaned_captions


# ==========================================================
# 5. VERIFICATION DES CAPTIONS
# ==========================================================

def validate_captions(kb_ids, raw_captions, cleaned_captions):

    raw_counts = Counter(
        item["image_id"] for item in raw_captions
    )

    cleaned_counts = Counter(
        item["image_id"] for item in cleaned_captions
    )

    missing_images = [
        image_id
        for image_id in kb_ids
        if raw_counts.get(image_id, 0) == 0
    ]

    images_without_five_captions = {
        image_id: raw_counts.get(image_id, 0)
        for image_id in kb_ids
        if raw_counts.get(image_id, 0) != 5
    }

    # Vérifier que les deux versions correspondent
    same_count = len(raw_captions) == len(cleaned_captions)

    return {
        "knowledge_base_images": len(kb_ids),
        "raw_captions_count": len(raw_captions),
        "cleaned_captions_count": len(cleaned_captions),
        "missing_images": missing_images,
        "images_without_five_captions": (
            images_without_five_captions
        ),
        "raw_cleaned_counts_match": same_count
    }


# ==========================================================
# 6. SAUVEGARDE
# ==========================================================

def save_json(data, output_path):

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved: {output_path}")


# ==========================================================
# 7. PROGRAMME PRINCIPAL
# ==========================================================

def main():

    print("=" * 55)
    print("COCO CAPTION PREPROCESSING")
    print("=" * 55)

    captions_data = load_json(CAPTIONS_FILE)
    kb_ids = load_json(KB_IDS_FILE)

    print(f"Knowledge Base images: {len(kb_ids)}")

    raw_captions, cleaned_captions = select_kb_captions(
        captions_data,
        kb_ids
    )

    statistics = validate_captions(
        kb_ids,
        raw_captions,
        cleaned_captions
    )

    # Sauvegarder les deux versions
    save_json(raw_captions, RAW_OUTPUT)
    save_json(cleaned_captions, CLEAN_OUTPUT)

    # Sauvegarder le rapport
    report_path = REPORT_DIR / "caption_preprocessing_report.json"

    save_json(statistics, report_path)

    print("\n========== RESULTS ==========")

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\nCAPTION PREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()
