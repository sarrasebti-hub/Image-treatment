
from pathlib import Path
import json

from PIL import Image, ImageOps
from tqdm import tqdm


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

SOURCE_DIR = DATA_DIR / "train2017"
SPLITS_DIR = DATA_DIR / "splits"

OUTPUT_DIR = DATA_DIR / "processed_images"

KB_IDS_FILE = SPLITS_DIR / "knowledge_base_image_ids.json"

REPORTS_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CHARGER LES IDS DE LA KNOWLEDGE BASE
# ==========================================================

def load_kb_ids():

    if not KB_IDS_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {KB_IDS_FILE}"
        )

    with open(KB_IDS_FILE, "r", encoding="utf-8") as file:
        image_ids = json.load(file)

    if not image_ids:
        raise ValueError("Knowledge Base is empty.")

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs detected.")

    return image_ids


# ==========================================================
# 3. PREPROCESSING D'UNE IMAGE
# ==========================================================

def preprocess_image(image_id):

    filename = f"{image_id:012d}.jpg"

    source_path = SOURCE_DIR / filename
    output_path = OUTPUT_DIR / filename

    if not source_path.exists():
        raise FileNotFoundError(
            f"Image not found: {source_path}"
        )

    with Image.open(source_path) as image:

        # Dimensions originales
        original_width, original_height = image.size

        # Correction de l'orientation EXIF
        image = ImageOps.exif_transpose(image)

        # Conversion RGB
        image = image.convert("RGB")

        # Dimensions après correction
        processed_width, processed_height = image.size

        # Sauvegarde
        image.save(
            output_path,
            format="JPEG",
            quality=95
        )

    return {
        "image_id": image_id,
        "filename": filename,
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "mode": "RGB",
        "status": "success"
    }


# ==========================================================
# 4. TRAITER TOUTES LES IMAGES
# ==========================================================

def process_images(image_ids):

    successful_images = []
    failed_images = []

    for image_id in tqdm(
        image_ids,
        desc="Image preprocessing"
    ):

        try:

            result = preprocess_image(image_id)

            successful_images.append(result)

        except Exception as error:

            failed_images.append({
                "image_id": image_id,
                "error": str(error)
            })

    return successful_images, failed_images


# ==========================================================
# 5. GENERER LE RAPPORT
# ==========================================================

def save_report(successful_images, failed_images):

    report = {
        "dataset": "COCO train2017",
        "split": "Knowledge Base",
        "total_images": (
            len(successful_images) + len(failed_images)
        ),
        "successful_images": len(successful_images),
        "failed_images": len(failed_images),
        "operations": [
            "EXIF orientation correction",
            "RGB conversion",
            "Original aspect ratio preservation"
        ],
        "images": successful_images,
        "errors": failed_images
    }

    report_path = (
        REPORTS_DIR / "image_preprocessing_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nReport saved: {report_path}")


# ==========================================================
# 6. PROGRAMME PRINCIPAL
# ==========================================================

def main():

    print("=" * 55)
    print("COCO IMAGE PREPROCESSING")
    print("=" * 55)

    image_ids = load_kb_ids()

    print(f"Images to process: {len(image_ids)}")

    successful, failed = process_images(image_ids)

    save_report(successful, failed)

    print("\nPREPROCESSING COMPLETED")
    print(f"Successful images: {len(successful)}")
    print(f"Failed images: {len(failed)}")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
