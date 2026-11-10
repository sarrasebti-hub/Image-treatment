
import json
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

KNOWLEDGE_BASE_PATH = (
    BASE_DIR / "splits" / "knowledge_base" / "knowledge_base_cleaned.json"
)

INSTANCES_PATH = (
    PROJECT_DIR / "annotations" / "instances_train2017.json"
)

OUTPUT_DIR = BASE_DIR / "metadata"
OUTPUT_PATH = OUTPUT_DIR / "knowledge_base_metadata.json"

# ==========================================
# 2. VERIFICATION DES FICHIERS
# ==========================================

for path in [KNOWLEDGE_BASE_PATH, INSTANCES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

print("Fichiers sources verifies.")

# ==========================================
# 3. CHARGEMENT DE LA KNOWLEDGE BASE
# ==========================================

print("Chargement de la Knowledge Base...")

with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
    kb_data = json.load(f)

selected_ids = {
    image["id"] for image in kb_data["images"]
}

print("Images selectionnees :", len(selected_ids))
print("Captions disponibles :", len(kb_data["annotations"]))

# ==========================================
# 4. CHARGEMENT DES ANNOTATIONS COCO
# ==========================================

print("Chargement des annotations COCO...")

with open(INSTANCES_PATH, "r", encoding="utf-8") as f:
    instances_data = json.load(f)

# ==========================================
# 5. MAPPING DES CATEGORIES
# ==========================================

categories = {
    category["id"]: category["name"]
    for category in instances_data["categories"]
}

# ==========================================
# 6. EXTRACTION DES METADONNEES IMAGES
# ==========================================

metadata = {}

for img in instances_data["images"]:
    image_id = img["id"]

    if image_id in selected_ids:
        metadata[image_id] = {
            "image_id": image_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "objects": [],
            "captions": []
        }

# ==========================================
# 7. EXTRACTION DES OBJETS
# ==========================================

for annotation in instances_data["annotations"]:
    image_id = annotation["image_id"]

    if image_id in metadata:
        category_name = categories.get(
            annotation["category_id"], "unknown"
        )

        metadata[image_id]["objects"].append({
            "category": category_name,
            "bbox": annotation["bbox"],
            "area": annotation["area"],
            "iscrowd": annotation["iscrowd"]
        })

# ==========================================
# 8. EXTRACTION DES CAPTIONS
# ==========================================

for annotation in kb_data["annotations"]:
    image_id = annotation["image_id"]

    if image_id in metadata:
        metadata[image_id]["captions"].append(
            annotation["caption"]
        )

# ==========================================
# 9. SAUVEGARDE DES METADONNEES
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata_list = list(metadata.values())

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        metadata_list,
        f,
        ensure_ascii=False,
        indent=2
    )

# ==========================================
# 10. RAPPORT FINAL
# ==========================================

images_without_metadata = selected_ids - set(metadata.keys())

images_without_captions = [
    item["image_id"]
    for item in metadata_list
    if len(item["captions"]) == 0
]

wrong_caption_count = [
    item["image_id"]
    for item in metadata_list
    if len(item["captions"]) != 5
]

print("\n========== RAPPORT FINAL ==========")

print("Images selectionnees :", len(selected_ids))
print("Images extraites :", len(metadata_list))
print("Images sans metadonnees :", len(images_without_metadata))
print("Images sans captions :", len(images_without_captions))
print("Images avec un nombre de captions different de 5 :",
      len(wrong_caption_count))

print("Fichier genere :", OUTPUT_PATH)

if (
    len(metadata_list) == len(selected_ids)
    and not images_without_metadata
    and not images_without_captions
    and not wrong_caption_count
):
    print("\nSUCCES : extraction validee.")
else:
    print("\nATTENTION : verification supplementaire necessaire.")
