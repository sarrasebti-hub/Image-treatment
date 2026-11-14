
import json
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "dataset" / "images_val"
SPLITS_PATH = BASE_DIR / "dataset" / "dataset_splits.json"

OUTPUT_DIR = BASE_DIR / "feature_extraction" / "results"
OUTPUT_PATH = OUTPUT_DIR / "test_clip_embeddings.pt"

# Model settings: same as Train and Validation
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

DEVICE = "cpu"
BATCH_SIZE = 32


def main():
    print("=" * 60)
    print("CLIP FEATURE EXTRACTION - TEST")
    print("=" * 60)

    # Load Test IDs
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = [int(image_id) for image_id in splits["test"]]

    print(f"Test images: {len(test_ids)}")

    # Verify images
    image_paths = []

    for image_id in test_ids:
        image_name = f"{image_id:012d}.jpg"
        image_path = IMAGES_DIR / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image_paths.append((image_id, image_name, image_path))

    print("All Test images were found.")

    # Load CLIP
    print(f"Loading CLIP: {MODEL_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    force_quick_gelu=True
)

    model = model.to(DEVICE)
    model.eval()

    all_embeddings = []
    image_ids = []
    image_names = []

    # Extract embeddings
    with torch.no_grad():

        for start in tqdm(
            range(0, len(image_paths), BATCH_SIZE),
            desc="Extracting CLIP features"
        ):
            batch = image_paths[start:start + BATCH_SIZE]

            images = []

            for image_id, image_name, image_path in batch:
                image = Image.open(image_path).convert("RGB")
                images.append(preprocess(image))

                image_ids.append(image_id)
                image_names.append(image_name)

            image_tensor = torch.stack(images).to(DEVICE)

            features = model.encode_image(image_tensor)

            # Normalize embeddings
            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

            all_embeddings.append(features.cpu())

    # Combine batches
    embeddings = torch.cat(all_embeddings, dim=0)

    # Save results using the same structure as Train/Validation
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "image_ids": torch.tensor(image_ids, dtype=torch.int64),
        "image_names": image_names,
        "embeddings": embeddings,
        "embedding_dimension": embeddings.shape[1],
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "normalized": True,
        "split": "test"
    }

    torch.save(results, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("TEST CLIP EXTRACTION COMPLETED")
    print("=" * 60)
    print(f"Images processed: {len(image_names)}")
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
