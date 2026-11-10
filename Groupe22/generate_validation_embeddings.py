
import json
import torch
import numpy as np

from pathlib import Path
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import normalize


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Fichiers Validation
SPLIT_FILE = (
    BASE_DIR
    / "dataset"
    / "splits"
    / "validation"
    / "validation_cleaned.json"
)

# Images originales Validation
IMAGE_DIR = BASE_DIR / "val2017"

# Dossier de sauvegarde
OUTPUT_DIR = BASE_DIR / "embeddings"

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cpu"

BATCH_SIZE = 8
EXPECTED_IMAGES = 100
EXPECTED_DIMENSION = 512


# ==================================================
# 2. VERIFICATION DES CHEMINS
# ==================================================

print("========== VERIFICATION DES CHEMINS ==========")

for path in [SPLIT_FILE, IMAGE_DIR]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chemin introuvable : {path}"
        )

print("Fichier Validation :", SPLIT_FILE)
print("Dossier images :", IMAGE_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. CHARGEMENT DES DONNEES
# ==================================================

print("\nChargement des donnees Validation...")

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict):
    raise ValueError("Format JSON incorrect.")

if "images" not in data:
    raise ValueError("La cle 'images' est absente.")

records = data["images"]

if len(records) != EXPECTED_IMAGES:
    raise ValueError(
        f"Nombre d'images incorrect : {len(records)}"
    )

image_ids = [item["id"] for item in records]

if len(set(image_ids)) != len(image_ids):
    raise ValueError("Des identifiants images sont dupliques.")

print("Nombre d'images :", len(records))
print("Identifiants uniques :", len(set(image_ids)))


# ==================================================
# 4. VERIFICATION DES IMAGES
# ==================================================

print("\nVerification des images...")

missing_images = []

for item in records:

    image_path = IMAGE_DIR / item["file_name"]

    if not image_path.is_file():
        missing_images.append(item["file_name"])

if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} images manquantes. "
        f"Exemples : {missing_images[:5]}"
    )

print("Toutes les images sont presentes.")


# ==================================================
# 5. CHARGEMENT DU MODELE CLIP
# ==================================================

print("\nChargement du modele CLIP...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model = CLIPModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("Modele charge.")
print("Device :", DEVICE)
print("Dimension CLIP :", model.config.projection_dim)

if model.config.projection_dim != EXPECTED_DIMENSION:
    raise ValueError("Dimension du modele CLIP incorrecte.")


# ==================================================
# 6. GENERATION DES EMBEDDINGS IMAGE
# ==================================================

print("\nGeneration des embeddings Validation...")

all_embeddings = []

with torch.inference_mode():

    for start in range(0, len(records), BATCH_SIZE):

        batch_records = records[
            start:start + BATCH_SIZE
        ]

        images = []

        for item in batch_records:

            image_path = IMAGE_DIR / item["file_name"]

            with Image.open(image_path) as img:

                img = ImageOps.exif_transpose(img)

                images.append(
                    img.convert("RGB")
                )

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Extraction des caracteristiques visuelles
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Projection dans l'espace CLIP
        features = model.visual_projection(
            vision_outputs.pooler_output
        )

        if features.shape[-1] != EXPECTED_DIMENSION:
            raise ValueError(
                f"Dimension incorrecte : {features.shape}"
            )

        # Normalisation L2
        embeddings = normalize(
            features,
            p=2,
            dim=1
        )

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        print(
            "Images traitees :",
            min(start + BATCH_SIZE, len(records)),
            "/",
            len(records)
        )


# ==================================================
# 7. CONVERSION ET VERIFICATION
# ==================================================

embeddings = np.concatenate(
    all_embeddings,
    axis=0
).astype(np.float32)

print("\n========== CONTROLE QUALITE ==========")

print("Dimensions :", embeddings.shape)
print("Type numerique :", embeddings.dtype)

if embeddings.shape != (EXPECTED_IMAGES, EXPECTED_DIMENSION):
    raise ValueError("Dimensions finales incorrectes.")

if not np.isfinite(embeddings).all():
    raise ValueError("Les embeddings contiennent des valeurs invalides.")

norms = np.linalg.norm(embeddings, axis=1)

if not np.allclose(norms, 1.0, atol=1e-4):
    raise ValueError("Normalisation L2 incorrecte.")

print("Valeurs numeriques : OK")
print("Normalisation L2 : OK")
print("Nombre IDs :", len(image_ids))
print("Correspondance IDs / embeddings : OK")


# ==================================================
# 8. SAUVEGARDE DES FICHIERS
# ==================================================

EMBEDDINGS_FILE = OUTPUT_DIR / "val2017_embeddings.npy"
IDS_FILE = OUTPUT_DIR / "val2017_ids.json"

np.save(
    EMBEDDINGS_FILE,
    embeddings
)

with open(IDS_FILE, "w", encoding="utf-8") as f:

    json.dump(
        image_ids,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==================================================
# 9. VERIFICATION APRES SAUVEGARDE
# ==================================================

saved_embeddings = np.load(EMBEDDINGS_FILE)

with open(IDS_FILE, "r", encoding="utf-8") as f:
    saved_ids = json.load(f)

if saved_embeddings.shape != (100, 512):
    raise ValueError("Erreur apres sauvegarde des embeddings.")

if saved_ids != image_ids:
    raise ValueError("Erreur apres sauvegarde des identifiants.")

print("\n========== RAPPORT FINAL ==========")

print("Images Validation :", len(records))
print("Embeddings :", saved_embeddings.shape)
print("Identifiants :", len(saved_ids))
print("Fichier embeddings :", EMBEDDINGS_FILE)
print("Fichier IDs :", IDS_FILE)

print("\nSUCCES : VALIDATION EMBEDDINGS GENERES.")
