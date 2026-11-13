
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_DIR / "data" / "splits"
CAPTIONS_DIR = PROJECT_DIR / "data" / "captions"
IMAGES_DIR = PROJECT_DIR / "data" / "processed_images"
EMBEDDINGS_DIR = PROJECT_DIR / "data" / "embeddings"

KB_IDS_FILE = SPLITS_DIR / "knowledge_base_image_ids.json"
CAPTIONS_FILE = CAPTIONS_DIR / "kb_captions_cleaned.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 32
MAX_TEXT_LENGTH = 77

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. UTILITAIRES JSON
# ============================================================

def load_json(file_path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_image_ids(data):
    """Extract image IDs from different JSON structures."""

    if isinstance(data, list):
        values = data

    elif isinstance(data, dict):
        values = data.get(
            "image_ids",
            data.get("knowledge_base_image_ids", data.get("ids"))
        )

        if values is None:
            raise ValueError(
                "Cannot find image IDs in the split JSON file."
            )

    else:
        raise ValueError("Unsupported image ID JSON format.")

    result = []

    for item in values:
        if isinstance(item, dict):
            item = item.get("image_id", item.get("id"))

        if item is not None:
            result.append(int(item))

    return result


# ============================================================
# 3. EXTRACTION DES CAPTIONS
# ============================================================

def extract_captions(data):
    """
    Extract captions from the cleaned captions JSON.

    Supported formats:
    - List of caption dictionaries
    - Dictionary containing a captions list
    - Dictionary containing image records
    """

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):

        if isinstance(data.get("captions"), list):
            records = data["captions"]

        elif isinstance(data.get("annotations"), list):
            records = data["annotations"]

        elif isinstance(data.get("images"), list):
            records = []

            for image in data["images"]:
                image_id = image.get("image_id", image.get("id"))

                captions = image.get(
                    "captions",
                    image.get("cleaned_captions", [])
                )

                for caption in captions:
                    if isinstance(caption, dict):
                        record = caption.copy()
                        record.setdefault("image_id", image_id)
                    else:
                        record = {
                            "image_id": image_id,
                            "caption": caption
                        }

                    records.append(record)

        else:
            raise ValueError(
                "Cannot identify captions in the cleaned JSON file. "
                "Expected captions, annotations or images."
            )

    else:
        raise ValueError("Unsupported captions JSON format.")

    extracted = []

    for index, item in enumerate(records):

        if not isinstance(item, dict):
            raise ValueError(
                f"Caption record {index} is not a dictionary."
            )

        image_id = item.get("image_id", item.get("id"))

        caption = item.get(
            "cleaned_caption",
            item.get("caption", item.get("text"))
        )

        annotation_id = item.get(
            "annotation_id",
            item.get("id", index)
        )

        if image_id is None or caption is None:
            raise ValueError(
                f"Missing image_id or caption in record {index}."
            )

        extracted.append({
            "image_id": int(image_id),
            "annotation_id": int(annotation_id),
            "caption": str(caption).strip()
        })

    return extracted


# ============================================================
# 4. FIND PROCESSED IMAGE
# ============================================================

def find_image(image_id):
    """Find a processed COCO image using its numeric ID."""

    filename = f"{image_id:012d}"

    possible_extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]

    for extension in possible_extensions:

        image_path = IMAGES_DIR / (filename + extension)

        if image_path.exists():
            return image_path

    raise FileNotFoundError(
        f"Processed image not found for image ID {image_id} "
        f"in {IMAGES_DIR}"
    )


# ============================================================
# 5. IMAGE EMBEDDINGS
# ============================================================

def generate_image_embeddings(
    model,
    processor,
    image_ids
):
    """Generate normalized CLIP embeddings for images."""

    all_embeddings = []
    successful_ids = []

    print("\nGenerating image embeddings...")

    for start in tqdm(
        range(0, len(image_ids), BATCH_SIZE),
        desc="Image embeddings"
    ):

        batch_ids = image_ids[
            start:start + BATCH_SIZE
        ]

        images = []
        valid_ids = []

        for image_id in batch_ids:

            image_path = find_image(image_id)

            try:
                image = Image.open(image_path).convert("RGB")

                images.append(image)
                valid_ids.append(image_id)

            except Exception as error:
                print(
                    f"\nWarning: Cannot read image "
                    f"{image_id}: {error}"
                )

        if not images:
            continue

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():

            # Compatible with Transformers versions
            # where get_image_features returns a model output.

            vision_outputs = model.vision_model(
                pixel_values=pixel_values
            )

            pooled_output = vision_outputs.pooler_output

            features = model.visual_projection(
                pooled_output
            )

            # L2 normalization

            features = features / features.norm(
                p=2,
                dim=-1,
                keepdim=True
            ).clamp(min=1e-12)

        all_embeddings.append(
            features.cpu().numpy()
        )

        successful_ids.extend(valid_ids)

    if not all_embeddings:
        raise RuntimeError(
            "No image embeddings were generated."
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    ids = np.array(
        successful_ids,
        dtype=np.int64
    )

    return embeddings, ids


# ============================================================
# 6. TEXT EMBEDDINGS
# ============================================================

def generate_text_embeddings(
    model,
    processor,
    captions
):
    """Generate normalized CLIP embeddings for captions."""

    all_embeddings = []
    image_ids = []
    annotation_ids = []

    caption_texts = [
        item["caption"] for item in captions
    ]

    print("\nGenerating text embeddings...")

    for start in tqdm(
        range(0, len(caption_texts), BATCH_SIZE),
        desc="Text embeddings"
    ):

        batch_texts = caption_texts[
            start:start + BATCH_SIZE
        ]

        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH
        )

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():

            # Compatible with Transformers versions
            # where get_text_features returns a model output.

            text_outputs = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            pooled_output = text_outputs.pooler_output

            features = model.text_projection(
                pooled_output
            )

            # L2 normalization

            features = features / features.norm(
                p=2,
                dim=-1,
                keepdim=True
            ).clamp(min=1e-12)

        all_embeddings.append(
            features.cpu().numpy()
        )

        batch_records = captions[
            start:start + BATCH_SIZE
        ]

        image_ids.extend([
            item["image_id"] for item in batch_records
        ])

        annotation_ids.extend([
            item["annotation_id"] for item in batch_records
        ])

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    return (
        embeddings,
        np.array(image_ids, dtype=np.int64),
        np.array(annotation_ids, dtype=np.int64)
    )


# ============================================================
# 7. SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    image_embeddings,
    image_ids,
    text_embeddings,
    text_image_ids,
    text_annotation_ids,
    metadata
):
    """Save all generated embeddings and metadata."""

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_DIR / "image_embeddings.npy",
        image_embeddings
    )

    np.save(
        EMBEDDINGS_DIR / "image_ids.npy",
        image_ids
    )

    np.save(
        EMBEDDINGS_DIR / "text_embeddings.npy",
        text_embeddings
    )

    np.save(
        EMBEDDINGS_DIR / "text_image_ids.npy",
        text_image_ids
    )

    np.save(
        EMBEDDINGS_DIR / "text_annotation_ids.npy",
        text_annotation_ids
    )

    with open(
        EMBEDDINGS_DIR / "embedding_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nAll embedding files saved successfully.")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 55)
    print("COCO EMBEDDING GENERATION")
    print("=" * 55)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    # Load Knowledge Base IDs

    kb_data = load_json(KB_IDS_FILE)

    kb_ids = extract_image_ids(kb_data)

    print(f"\nKnowledge Base images: {len(kb_ids)}")

    # Load cleaned captions

    captions_data = load_json(CAPTIONS_FILE)

    captions = extract_captions(captions_data)

    # Keep only captions belonging to the Knowledge Base

    kb_id_set = set(kb_ids)

    captions = [
        item for item in captions
        if item["image_id"] in kb_id_set
    ]

    print(f"Captions: {len(captions)}")

    # Check that all KB images have captions

    caption_image_ids = {
        item["image_id"] for item in captions
    }

    missing_caption_ids = kb_id_set - caption_image_ids

    if missing_caption_ids:
        raise ValueError(
            f"{len(missing_caption_ids)} Knowledge Base images "
            "have no captions."
        )

    # Check processed images before loading the model

    missing_images = []

    for image_id in kb_ids:

        try:
            find_image(image_id)

        except FileNotFoundError:
            missing_images.append(image_id)

    if missing_images:

        raise FileNotFoundError(
            f"{len(missing_images)} processed images are missing. "
            f"First missing IDs: {missing_images[:10]}. "
            "Run preprocess_images.py first."
        )

    # Load CLIP model

    print("\nLoading CLIP model...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("CLIP model loaded successfully.")

    # Generate image embeddings

    image_embeddings, image_ids = generate_image_embeddings(
        model,
        processor,
        kb_ids
    )

    # Generate text embeddings

    (
        text_embeddings,
        text_image_ids,
        text_annotation_ids
    ) = generate_text_embeddings(
        model,
        processor,
        captions
    )

    # Check embedding dimensions

    print("\nChecking embedding dimensions...")

    print("Image embeddings:", image_embeddings.shape)
    print("Text embeddings:", text_embeddings.shape)

    if image_embeddings.shape[1] != 512:
        raise ValueError(
            "Unexpected image embedding dimension."
        )

    if text_embeddings.shape[1] != 512:
        raise ValueError(
            "Unexpected text embedding dimension."
        )

    # Save results

    metadata = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "embedding_dimension": 512,
        "number_of_images": int(len(image_ids)),
        "number_of_captions": int(len(text_image_ids)),
        "batch_size": BATCH_SIZE,
        "image_embeddings_shape": list(
            image_embeddings.shape
        ),
        "text_embeddings_shape": list(
            text_embeddings.shape
        ),
        "images_normalized": True,
        "texts_normalized": True
    }

    save_embeddings(
        image_embeddings,
        image_ids,
        text_embeddings,
        text_image_ids,
        text_annotation_ids,
        metadata
    )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 55)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 55)

    print(f"Images processed: {len(image_ids)}")
    print(f"Captions processed: {len(text_image_ids)}")

    print(
        f"Image embeddings shape: "
        f"{image_embeddings.shape}"
    )

    print(
        f"Text embeddings shape: "
        f"{text_embeddings.shape}"
    )

    print(
        f"Execution time: {elapsed_time / 60:.2f} minutes"
    )

    print(f"\nOutput directory: {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
