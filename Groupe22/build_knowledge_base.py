
import json
import re
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. CHEMINS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"

ANNOTATIONS_FILE = DATASET_DIR / "annotations" / "captions_train2017.json"
TRAIN_IDS_FILE = DATASET_DIR / "embeddings" / "train2017_ids.json"
OUTPUT_FILE = DATASET_DIR / "knowledge_base_train.json"

print("Debut de la construction de la base...")

# ==========================================
# 2. VERIFICATION DES FICHIERS
# ==========================================

for path in [ANNOTATIONS_FILE, TRAIN_IDS_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

# ==========================================
# 3. CHARGEMENT DES IDS TRAIN
# ==========================================

with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
    train_ids_data = json.load(f)

if isinstance(train_ids_data, dict):
    train_ids = train_ids_data.get("ids")
    if train_ids is None:
        raise ValueError("Le dictionnaire ne contient pas la cle 'ids'.")
else:
    train_ids = train_ids_data

train_ids = {
    int(Path(str(image_id)).stem)
    for image_id in train_ids
}

print(f"Images train chargees : {len(train_ids)}")

# ==========================================
# 4. CHARGEMENT DES ANNOTATIONS COCO
# ==========================================

print("Chargement des annotations COCO...")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# ==========================================
# 5. REGROUPEMENT DES LEGENDES
# ==========================================

captions_by_image = defaultdict(list)

for annotation in coco_data["annotations"]:
    image_id = int(annotation["image_id"])
    caption = annotation["caption"].strip()

    if caption:
        captions_by_image[image_id].append(caption)

# ==========================================
# 6. CONSTRUCTION DE LA BASE
# ==========================================

knowledge_base = []
missing_images = []
wrong_caption_count = []

for image_id in sorted(train_ids):

    captions = captions_by_image.get(image_id, [])

    if len(captions) == 0:
        missing_images.append(image_id)
        continue

    if len(captions) != 5:
        wrong_caption_count.append({
            "image_id": image_id,
            "caption_count": len(captions)
        })

    image_filename = f"{image_id:012d}.jpg"

    cleaned_captions = [
        re.sub(r"\s+", " ", caption.lower()).strip()
        for caption in captions
    ]

    knowledge_base.append({
        "image_id": image_id,
        "image_filename": image_filename,
        "captions": captions,
        "cleaned_captions": cleaned_captions
    })

# ==========================================
# 7. SAUVEGARDE
# ==========================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        knowledge_base,
        f,
        ensure_ascii=False,
        indent=2
    )

# ==========================================
# 8. RAPPORT FINAL
# ==========================================

print("\n========== RAPPORT FINAL ==========")
print(f"Images train demandees : {len(train_ids)}")
print(f"Images enregistrees : {len(knowledge_base)}")
print(f"Images sans legende : {len(missing_images)}")
print(f"Images avec un nombre de legendes different de 5 : {len(wrong_caption_count)}")

if knowledge_base:
    print("\nExemple de la premiere image :")
    print(json.dumps(
        knowledge_base[0],
        ensure_ascii=False,
        indent=2
    ))

print(f"\nBase sauvegardee : {OUTPUT_FILE}")

if missing_images:
    print("Premiers IDs sans legende :", missing_images[:10])

if wrong_caption_count:
    print("Premiers cas avec un nombre incorrect de legendes :")
    print(wrong_caption_count[:10])

if (
    len(knowledge_base) == len(train_ids)
    and not missing_images
    and not wrong_caption_count
):
    print("\nSUCCES : toutes les images ont exactement 5 legendes.")
else:
    print("\nATTENTION : verification necessaire avant de continuer.")
