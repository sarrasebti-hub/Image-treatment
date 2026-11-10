
import json
import torch
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import normalize


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

METADATA_PATH = BASE_DIR / "metadata" / "knowledge_base_metadata.json"
IMAGE_DIR = PROJECT_DIR / "train2017"
OUTPUT_DIR = BASE_DIR / "embeddings"

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cpu"

IMAGE_BATCH_SIZE = 8
TEXT_BATCH_SIZE = 8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. VERIFICATION DES FICHIERS
# ==========================================

for path in [METADATA_PATH, IMAGE_DIR]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier ou dossier introuvable : {path}"
        )

print("Chemins verifies.")


# ==========================================
# 3. CHARGEMENT DES METADONNEES
# ==========================================

print("\nChargement des metadonnees...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

if not isinstance(metadata, list):
    raise ValueError("Les metadonnees doivent etre une liste.")

if len(metadata) != 10000:
    raise ValueError(
        f"Nombre d'images incorrect : {len(metadata)}"
    )

all_captions = []
caption_records = []

for item in metadata:

    if len(item["captions"]) != 5:
        raise ValueError(
            f"Image {item['image_id']} : nombre de captions incorrect."
        )

    for caption in item["captions"]:

        all_captions.append(caption)

        caption_records.append({
            "image_id": item["image_id"],
            "caption": caption
        })

print("Images :", len(metadata))
print("Captions :", len(all_captions))


# ==========================================
# 4. VERIFICATION DES IMAGES
# ==========================================

print("\nVerification des images...")

missing_images = []

for item in metadata:

    image_path = IMAGE_DIR / item["file_name"]

    if not image_path.is_file():
        missing_images.append(item["file_name"])

if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} images introuvables. "
        f"Exemples : {missing_images[:5]}"
    )

print("Toutes les images sont presentes.")


# ==========================================
# 5. CHARGEMENT DE CLIP
# ==========================================

print("\nChargement du modele CLIP...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("Modele charge.")
print("Device :", DEVICE)
print("Dimension attendue :", model.config.projection_dim)


# ==========================================
# 6. EMBEDDINGS DES IMAGES
# ==========================================

image_embeddings = []
image_ids = []

print("\nGeneration des embeddings image...")

with torch.inference_mode():

    for start in range(0, len(metadata), IMAGE_BATCH_SIZE):

        batch_records = metadata[
            start:start + IMAGE_BATCH_SIZE
        ]

        images = []

        for item in batch_records:

            image_path = IMAGE_DIR / item["file_name"]

            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                images.append(img.convert("RGB"))

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

        if features.shape[-1] != model.config.projection_dim:
            raise ValueError(
                f"Dimension image incorrecte : {features.shape}"
            )

        # Normalisation L2
        embeddings = normalize(
            features,
            p=2,
            dim=1
        )

        image_embeddings.extend(
            embeddings.cpu().numpy()
        )

        image_ids.extend(
            item["image_id"] for item in batch_records
        )

        print(
            f"Images traitees : "
            f"{min(start + IMAGE_BATCH_SIZE, len(metadata))}"
            f"/{len(metadata)}"
        )


image_embeddings = np.asarray(
    image_embeddings,
    dtype=np.float32
)


# Sauvegarde des embeddings image

np.save(
    OUTPUT_DIR / "image_embeddings.npy",
    image_embeddings
)

with open(
    OUTPUT_DIR / "image_ids.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(image_ids, f, ensure_ascii=False)

print("Embeddings image sauvegardes.")


# ==========================================
# 7. EMBEDDINGS DES CAPTIONS
# ==========================================

text_embeddings = []

print("\nGeneration des embeddings texte...")

with torch.inference_mode():

    for start in range(
        0,
        len(all_captions),
        TEXT_BATCH_SIZE
    ):

        batch = all_captions[
            start:start + TEXT_BATCH_SIZE
        ]

        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Extraction des caracteristiques textuelles
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )

        # Projection dans l'espace CLIP
        features = model.text_projection(
            text_outputs.pooler_output
        )

        if features.shape[-1] != model.config.projection_dim:
            raise ValueError(
                f"Dimension texte incorrecte : {features.shape}"
            )

        # Normalisation L2
        embeddings = normalize(
            features,
            p=2,
            dim=1
        )

        text_embeddings.extend(
            embeddings.cpu().numpy()
        )

        print(
            f"Captions traitees : "
            f"{min(start + TEXT_BATCH_SIZE, len(all_captions))}"
            f"/{len(all_captions)}"
        )


text_embeddings = np.asarray(
    text_embeddings,
    dtype=np.float32
)


# Sauvegarde des embeddings texte

np.save(
    OUTPUT_DIR / "text_embeddings.npy",
    text_embeddings
)

with open(
    OUTPUT_DIR / "caption_records.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        caption_records,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Embeddings texte sauvegardes.")


# ==========================================
# 8. VERIFICATION FINALE
# ==========================================

print("\n========== VERIFICATION FINALE ==========")

print("Image embeddings :", image_embeddings.shape)
print("Text embeddings :", text_embeddings.shape)

print(
    "Embeddings image valides :",
    np.isfinite(image_embeddings).all()
)

print(
    "Embeddings texte valides :",
    np.isfinite(text_embeddings).all()
)

print(
    "Normes image correctes :",
    np.allclose(
        np.linalg.norm(image_embeddings, axis=1),
        1.0,
        atol=1e-4
    )
)

print(
    "Normes texte correctes :",
    np.allclose(
        np.linalg.norm(text_embeddings, axis=1),
        1.0,
        atol=1e-4
    )
)

print(
    "Correspondance images :",
    len(image_ids) == len(metadata)
)

print(
    "Correspondance captions :",
    len(caption_records) == len(text_embeddings)
)

print("\nFichiers generes :")

for file in OUTPUT_DIR.iterdir():

    if file.is_file():

        size_mb = round(
            file.stat().st_size / (1024 * 1024),
            2
        )

        print(file.name, "-", size_mb, "MB")


print("\nGeneration terminee !")
