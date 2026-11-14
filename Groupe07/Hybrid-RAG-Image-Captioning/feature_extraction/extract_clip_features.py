
import json
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm


# ==========================================
# 1. Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "dataset" / "images_train"
SPLITS_PATH = BASE_DIR / "dataset" / "dataset_splits.json"

OUTPUT_DIR = BASE_DIR / "feature_extraction" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
DEVICE = "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


# ==========================================
# 2. Load dataset splits
# ==========================================

print("Loading dataset splits...")

with open(SPLITS_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)

train_ids = sorted(set(splits["train"]))
validation_ids = sorted(set(splits["validation"]))

print("Train images:", len(train_ids))
print("Validation images:", len(validation_ids))


# ==========================================
# 3. Verify image files
# ==========================================

all_ids = train_ids + validation_ids

missing_ids = [
    image_id
    for image_id in all_ids
    if not (IMAGES_DIR / f"{image_id:012d}.jpg").exists()
]

if missing_ids:
    raise FileNotFoundError(
        f"{len(missing_ids)} images are missing. "
        f"First missing IDs: {missing_ids[:10]}"
    )

print("All selected image files exist.")


# ==========================================
# 4. Load CLIP model
# ==========================================

print("\nLoading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    force_quick_gelu=True
)

model = model.to(DEVICE)
model.eval()

print("CLIP model loaded successfully.")


# ==========================================
# 5. Extract image embeddings
# ==========================================

def extract_split(image_ids, split_name):

    print(f"\nStarting {split_name} extraction...")

    all_embeddings = []
    processed_ids = []

    for start in tqdm(
        range(0, len(image_ids), BATCH_SIZE),
        desc=f"CLIP {split_name}"
    ):

        batch_ids = image_ids[start:start + BATCH_SIZE]

        batch_images = []
        valid_ids = []

        for image_id in batch_ids:

            image_path = IMAGES_DIR / f"{image_id:012d}.jpg"

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image_tensor = preprocess(image)

                batch_images.append(image_tensor)
                valid_ids.append(image_id)

            except Exception as error:
                print(f"\nCould not process {image_path.name}: {error}")

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(DEVICE)

        with torch.no_grad():

            features = model.encode_image(batch_tensor)

            # Normalize embeddings
            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

        all_embeddings.append(features.cpu())
        processed_ids.extend(valid_ids)

    if not all_embeddings:
        raise RuntimeError(f"No embeddings generated for {split_name}.")

    embeddings = torch.cat(all_embeddings, dim=0)

    output_data = {
        "image_ids": torch.tensor(processed_ids, dtype=torch.int64),
        "image_names": [
            f"{image_id:012d}.jpg"
            for image_id in processed_ids
        ],
        "embeddings": embeddings,
        "embedding_dimension": embeddings.shape[1],
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "normalized": True,
        "split": split_name
    }

    output_path = OUTPUT_DIR / f"{split_name}_clip_embeddings.pt"

    torch.save(output_data, output_path)

    print(f"\n{split_name} completed!")
    print("Images processed:", len(processed_ids))
    print("Embedding shape:", tuple(embeddings.shape))
    print("Output:", output_path)


# ==========================================
# 6. Run extraction
# ==========================================

if __name__ == "__main__":

    extract_split(train_ids, "train")

    extract_split(validation_ids, "validation")

    print("\nCLIP FEATURE EXTRACTION COMPLETED!")
