"""
CLIP image embedding generation.

Generates image embeddings using:
openai/clip-vit-base-patch32

The generated embeddings are projected CLIP image features.
L2 normalization is applied to each image embedding.
The output embeddings are stored as float32.
"""
import json
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from data.preprocessing.image_preprocessing import (
    load_and_prepare_image
)


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def generate_clip_image_embeddings(
    image_records,
    image_dir,
    output_dir,
    batch_size=16,
    model_name=CLIP_MODEL_NAME,
    device=None,
    show_progress=True
):
    """
    Generate and save CLIP image embeddings.

    Parameters
    ----------
    image_records : list
        COCO image records containing 'id' and 'file_name'.

    image_dir : str or Path
        Directory containing the source images.

    output_dir : str or Path
        Directory where embeddings and mapping are saved.

    batch_size : int
        Number of images processed per batch.

    model_name : str
        Hugging Face CLIP model identifier.

    device : str or torch.device, optional
        Device used for inference. Automatically selected if omitted.

    show_progress : bool
        Whether to display a progress bar.

    Returns
    -------
    dict
        Summary of the embedding generation.
    """

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_records:
        raise ValueError("image_records cannot be empty.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    # Check image IDs and filenames
    image_ids = [record["id"] for record in image_records]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs found.")

    for record in image_records:
        if "id" not in record or "file_name" not in record:
            raise ValueError(
                "Every image record must contain 'id' and 'file_name'."
            )

    # Ensure deterministic row order
    image_records = sorted(
        image_records,
        key=lambda record: record["id"]
    )

    # Select device
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    print(f"Loading CLIP model: {model_name}")
    print(f"Device: {device}")

    # Load processor and model
    processor = CLIPProcessor.from_pretrained(model_name)

    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embedding_dimension = model.config.projection_dim

    embeddings_batches = []
    embedding_mapping = []

    # Process images in batches
    batch_starts = range(0, len(image_records), batch_size)

    if show_progress:
        batch_starts = tqdm(
            batch_starts,
            total=(len(image_records) + batch_size - 1) // batch_size,
            desc="Generating CLIP image embeddings"
        )

    for start in batch_starts:

        batch_records = image_records[start:start + batch_size]
        batch_images = []

        for record in batch_records:

            image_path = image_dir / record["file_name"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = load_and_prepare_image(image_path)
            batch_images.append(image)

        # Apply the official CLIP image processor
        inputs = processor(
            images=batch_images,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(device)

        # Extract projected image features

        with torch.inference_mode():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                return_dict=True
            )

            pooled_output = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled_output)

            # Apply L2 normalization to each image embedding
            image_features = torch.nn.functional.normalize(
                image_features,
                p=2,
                dim=-1
            )

        image_features = image_features.float().cpu().numpy()

        embeddings_batches.append(image_features)

        # Preserve the relationship between rows and images
        for record in batch_records:
            embedding_mapping.append({
                "row": len(embedding_mapping),
                "image_id": record["id"],
                "file_name": record["file_name"]
            })

        # Release temporary objects
        del batch_images, inputs, pixel_values, image_features

    # Combine batches
    embeddings = np.concatenate(
        embeddings_batches,
        axis=0
    ).astype(np.float32, copy=False)

    # Validate results before saving
    expected_shape = (
        len(image_records),
        embedding_dimension
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    if len(embedding_mapping) != len(image_records):
        raise ValueError(
            "Embedding mapping count does not match image count."
        )

    # Save artifacts
    embeddings_path = (
        output_dir / "clip_image_embeddings.npy"
    )

    mapping_path = (
        output_dir / "image_embedding_index.json"
    )

    np.save(embeddings_path, embeddings)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            embedding_mapping,
            f,
            ensure_ascii=False,
            indent=2
        )

    summary = {
        "model_name": model_name,
        "number_of_images": len(image_records),
        "embedding_dimension": embedding_dimension,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_norm": "L2",
        "normalized": True,
        "device": str(device),
        "batch_size": batch_size,
        "embeddings_file": str(embeddings_path),
        "mapping_file": str(mapping_path),
        "status": "success"
    }

    print("\nCLIP embedding generation completed.")
    print(f"Shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print(f"Embeddings saved: {embeddings_path}")
    print(f"Mapping saved: {mapping_path}")

    return summary
