
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

TEST_JSON = (
    DATASET_DIR
    / "dataset"
    / "splits"
    / "test"
    / "test_cleaned.json"
)

IMAGE_DIR = DATASET_DIR / "val2017"

MODEL_NAME = "openai/clip-vit-base-patch32"

OUTPUT_EMBEDDINGS = EMBEDDINGS_DIR / "test2017_embeddings.npy"
OUTPUT_IDS = EMBEDDINGS_DIR / "test2017_ids.json"

BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("=" * 60)
print("EXTRACTING TEST IMAGE EMBEDDINGS")
print("=" * 60)

with open(TEST_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]

if not images:
    raise ValueError("Test dataset is empty.")

image_ids = [
    int(item["id"])
    for item in images
]

if len(image_ids) != len(set(image_ids)):
    raise ValueError("Duplicate image IDs found in test dataset.")

image_paths = [
    IMAGE_DIR / item["file_name"]
    for item in images
]

missing = [
    str(path)
    for path in image_paths
    if not path.is_file()
]

if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. "
        f"Examples: {missing[:5]}"
    )

print("Test images:", len(images))
print("Unique IDs:", len(set(image_ids)))
print("Batch size:", BATCH_SIZE)
print("Device:", DEVICE)


# ============================================================
# 3. LOAD CLIP MODEL AND PROCESSOR
# ============================================================

print("\nLoading CLIP model...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model = CLIPModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("Model loaded:", MODEL_NAME)


# ============================================================
# 4. EXTRACT IMAGE EMBEDDINGS
# ============================================================

all_embeddings = []

print("\nExtracting embeddings...")

with torch.no_grad():

    for start in range(0, len(image_paths), BATCH_SIZE):

        batch_paths = image_paths[start:start + BATCH_SIZE]

        batch_images = []

        # Load images, correct EXIF orientation and convert to RGB
        for path in batch_paths:

            with Image.open(path) as img:

                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")

                batch_images.append(img.copy())

        try:

            inputs = processor(
                images=batch_images,
                return_tensors="pt"
            )

            pixel_values = inputs["pixel_values"].to(DEVICE)

            vision_outputs = model.vision_model(
                pixel_values=pixel_values
            )

            features = model.visual_projection(
                vision_outputs.pooler_output
            )

            # L2 normalization
            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=-1
            )

            batch_embeddings = (
                features.cpu().numpy().astype(np.float32)
            )

            all_embeddings.append(batch_embeddings)

        finally:

            # Close all loaded images
            for img in batch_images:
                img.close()

        processed = min(start + BATCH_SIZE, len(image_paths))

        print(f"Processed {processed}/{len(image_paths)}")


# ============================================================
# 5. BUILD EMBEDDING ARRAY
# ============================================================

embeddings = np.concatenate(
    all_embeddings,
    axis=0
)


# ============================================================
# 6. QUALITY CONTROL
# ============================================================

print("\nRunning quality checks...")

expected_shape = (len(images), 512)

if embeddings.shape != expected_shape:

    raise ValueError(
        f"Unexpected embedding shape: {embeddings.shape}. "
        f"Expected: {expected_shape}"
    )

if embeddings.dtype != np.float32:

    raise ValueError(
        f"Unexpected dtype: {embeddings.dtype}"
    )

if not np.isfinite(embeddings).all():

    raise ValueError(
        "Embeddings contain NaN or Inf values."
    )

norms = np.linalg.norm(
    embeddings,
    axis=1
)

if not np.allclose(norms, 1.0, atol=1e-5):

    raise ValueError(
        "Embeddings are not correctly L2 normalized."
    )

if len(image_ids) != embeddings.shape[0]:

    raise ValueError(
        "Mismatch between image IDs and embeddings."
    )

print("Shape:", embeddings.shape)
print("Dtype:", embeddings.dtype)
print("All finite:", np.isfinite(embeddings).all())
print("Mean L2 norm:", float(norms.mean()))
print("Min L2 norm:", float(norms.min()))
print("Max L2 norm:", float(norms.max()))


# ============================================================
# 7. SAVE RESULTS
# ============================================================

EMBEDDINGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT_EMBEDDINGS,
    embeddings
)

with open(OUTPUT_IDS, "w", encoding="utf-8") as f:

    json.dump(
        image_ids,
        f,
        indent=2
    )


# ============================================================
# 8. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("TEST EMBEDDING EXTRACTION COMPLETED")
print("=" * 60)

print("Model:", MODEL_NAME)
print("Device:", DEVICE)
print("Batch size:", BATCH_SIZE)

print("Number of images:", len(images))
print("Number of embeddings:", len(embeddings))

print("Embedding shape:", embeddings.shape)
print("Embedding dtype:", embeddings.dtype)

print("Unique image IDs:", len(set(image_ids)))

print("All embeddings valid:", np.isfinite(embeddings).all())

print("Mean L2 norm:", float(norms.mean()))

print("Embeddings saved:", OUTPUT_EMBEDDINGS)
print("IDs saved:", OUTPUT_IDS)

print("=" * 60)
