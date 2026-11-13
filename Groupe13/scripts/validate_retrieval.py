
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
INDEX_DIR = DATA_DIR / "index"
REPORTS_DIR = PROJECT_DIR / "reports"

MODEL_NAME = "openai/clip-vit-base-patch32"

KB_IDS_FILE = SPLITS_DIR / "knowledge_base_image_ids.json"
VALIDATION_IDS_FILE = SPLITS_DIR / "validation_image_ids.json"

CAPTIONS_FILE = ANNOTATIONS_DIR / "captions_train2017.json"

TRAIN_IMAGES_DIR = DATA_DIR / "train2017"

IMAGE_INDEX_FILE = INDEX_DIR / "image_index.faiss"
TEXT_INDEX_FILE = INDEX_DIR / "text_index.faiss"

IMAGE_INDEX_IDS_FILE = INDEX_DIR / "image_index_ids.npy"
TEXT_IMAGE_IDS_FILE = INDEX_DIR / "text_index_image_ids.npy"
TEXT_ANNOTATION_IDS_FILE = INDEX_DIR / "text_index_annotation_ids.npy"

OUTPUT_DIR = REPORTS_DIR / "validation"

TOP_K = 10
BATCH_SIZE = 16
EMBEDDING_DIMENSION = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_ids(data):

    if isinstance(data, list):
        values = data

    elif isinstance(data, dict):
        values = data.get(
            "image_ids",
            data.get(
                "knowledge_base_image_ids",
                data.get("ids")
            )
        )

        if values is None:
            raise ValueError("Cannot find image IDs in JSON.")

    else:
        raise ValueError("Unsupported split JSON format.")

    result = []

    for item in values:

        if isinstance(item, dict):
            item = item.get("image_id", item.get("id"))

        if item is not None:
            result.append(int(item))

    return result


def load_validation_data():

    print("\nLoading validation data...")

    kb_ids = extract_ids(load_json(KB_IDS_FILE))
    validation_ids = extract_ids(load_json(VALIDATION_IDS_FILE))

    kb_set = set(kb_ids)
    validation_set = set(validation_ids)

    if len(validation_set) != len(validation_ids):
        raise ValueError("Duplicate validation IDs detected.")

    overlap = kb_set & validation_set

    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} overlapping IDs."
        )

    coco_data = load_json(CAPTIONS_FILE)

    annotations = coco_data["annotations"]

    validation_captions = {}

    for annotation in annotations:

        image_id = int(annotation["image_id"])

        if image_id in validation_set:

            validation_captions.setdefault(
                image_id, []
            ).append({
                "annotation_id": int(annotation["id"]),
                "caption": annotation["caption"]
            })

    missing = [
        image_id
        for image_id in validation_ids
        if image_id not in validation_captions
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} validation images have no captions."
        )

    print("Knowledge Base images:", len(kb_ids))
    print("Validation images:", len(validation_ids))
    print(
        "Validation captions:",
        sum(len(v) for v in validation_captions.values())
    )

    return kb_ids, validation_ids, validation_captions


# ============================================================
# 3. LOAD FAISS INDEXES
# ============================================================

def load_indexes():

    print("\nLoading FAISS indexes...")

    required_files = [
        IMAGE_INDEX_FILE,
        TEXT_INDEX_FILE,
        IMAGE_INDEX_IDS_FILE,
        TEXT_IMAGE_IDS_FILE,
        TEXT_ANNOTATION_IDS_FILE
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required index file not found: {path}"
            )

    image_index = faiss.read_index(str(IMAGE_INDEX_FILE))
    text_index = faiss.read_index(str(TEXT_INDEX_FILE))

    image_index_ids = np.load(IMAGE_INDEX_IDS_FILE)
    text_image_ids = np.load(TEXT_IMAGE_IDS_FILE)
    text_annotation_ids = np.load(TEXT_ANNOTATION_IDS_FILE)

    if image_index.ntotal != len(image_index_ids):
        raise ValueError("Image index mapping mismatch.")

    if text_index.ntotal != len(text_image_ids):
        raise ValueError("Text index mapping mismatch.")

    if text_index.ntotal != len(text_annotation_ids):
        raise ValueError("Text annotation mapping mismatch.")

    if image_index.d != EMBEDDING_DIMENSION:
        raise ValueError("Invalid image index dimension.")

    if text_index.d != EMBEDDING_DIMENSION:
        raise ValueError("Invalid text index dimension.")

    print("Image index:", image_index.ntotal)
    print("Text index:", text_index.ntotal)

    return (
        image_index,
        text_index,
        image_index_ids,
        text_image_ids,
        text_annotation_ids
    )


# ============================================================
# 4. LOAD CLIP MODEL
# ============================================================

def load_model():

    print("\nLoading CLIP model...")
    print("Device:", DEVICE)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    print("CLIP model loaded successfully.")

    return model, processor


# ============================================================
# 5. GENERATE IMAGE EMBEDDINGS
# ============================================================

def encode_images(model, processor, image_ids):

    print("\nGenerating validation image embeddings...")

    embeddings = []

    for start in tqdm(
        range(0, len(image_ids), BATCH_SIZE),
        desc="Validation images"
    ):

        batch_ids = image_ids[start:start + BATCH_SIZE]

        images = []

        for image_id in batch_ids:

            image_path = (
                TRAIN_IMAGES_DIR /
                f"{image_id:012d}.jpg"
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Validation image not found: {image_path}"
                )

            with Image.open(image_path) as img:

                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                images.append(img.copy())

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():

            vision_output = model.vision_model(
                pixel_values=pixel_values
            )

            pooled = vision_output.pooler_output

            features = model.visual_projection(pooled)

            features = torch.nn.functional.normalize(
                features, p=2, dim=1
            )

        embeddings.append(
            features.cpu().numpy().astype(np.float32)
        )

    return np.concatenate(embeddings, axis=0)


# ============================================================
# 6. GENERATE TEXT EMBEDDINGS
# ============================================================

def encode_texts(model, processor, texts):

    print("\nGenerating validation caption embeddings...")

    embeddings = []

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Validation captions"
    ):

        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():

            text_output = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            pooled = text_output.pooler_output

            features = model.text_projection(pooled)

            features = torch.nn.functional.normalize(
                features, p=2, dim=1
            )

        embeddings.append(
            features.cpu().numpy().astype(np.float32)
        )

    return np.concatenate(embeddings, axis=0)


# ============================================================
# 7. SEARCH FAISS INDEX
# ============================================================

def search_index(index, query_embeddings, top_k):

    k = min(top_k, index.ntotal)

    scores, positions = index.search(
        np.ascontiguousarray(
            query_embeddings,
            dtype=np.float32
        ),
        k
    )

    return scores, positions


# ============================================================
# 8. FORMAT RESULTS
# ============================================================

def format_results(
    scores,
    positions,
    mapping_ids,
    annotation_ids=None
):

    results = []

    for query_scores, query_positions in zip(scores, positions):

        retrieved = []

        for score, position in zip(
            query_scores, query_positions
        ):

            if position < 0:
                continue

            item = {
                "similarity": round(float(score), 6),
                "image_id": int(mapping_ids[position])
            }

            if annotation_ids is not None:
                item["annotation_id"] = int(
                    annotation_ids[position]
                )

            retrieved.append(item)

        results.append(retrieved)

    return results


# ============================================================
# 9. RUN FOUR RETRIEVAL EXPERIMENTS
# ============================================================

def run_experiments(
    image_index,
    text_index,
    image_index_ids,
    text_image_ids,
    text_annotation_ids,
    validation_ids,
    validation_captions,
    image_embeddings,
    text_embeddings
):

    print("\nRunning retrieval experiments...")

    # Experiment 1:
    # Validation image -> Image index

    print("\n1. Image query -> Image index")

    image_image_scores, image_image_positions = search_index(
        image_index,
        image_embeddings,
        TOP_K
    )

    image_to_image = format_results(
        image_image_scores,
        image_image_positions,
        image_index_ids
    )

    # Experiment 2:
    # Validation image -> Text index

    print("2. Image query -> Text index")

    image_text_scores, image_text_positions = search_index(
        text_index,
        image_embeddings,
        TOP_K
    )

    image_to_text = format_results(
        image_text_scores,
        image_text_positions,
        text_image_ids,
        text_annotation_ids
    )

    # Experiment 3:
    # Validation caption -> Image index

    print("3. Caption query -> Image index")

    text_image_scores, text_image_positions = search_index(
        image_index,
        text_embeddings,
        TOP_K
    )

    text_to_image = format_results(
        text_image_scores,
        text_image_positions,
        image_index_ids
    )

    # Experiment 4:
    # Validation caption -> Text index

    print("4. Caption query -> Text index")

    text_text_scores, text_text_positions = search_index(
        text_index,
        text_embeddings,
        TOP_K
    )

    text_to_text = format_results(
        text_text_scores,
        text_text_positions,
        text_image_ids,
        text_annotation_ids
    )

    return {
        "image_to_image": image_to_image,
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "text_to_text": text_to_text
    }


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_results(
    validation_ids,
    validation_captions,
    experiments
):

    print("\nSaving validation results...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_records = []

    for i, image_id in enumerate(validation_ids):

        image_records.append({
            "query_image_id": int(image_id),
            "captions": validation_captions.get(image_id, []),
            "image_to_image": experiments["image_to_image"][i],
            "image_to_text": experiments["image_to_text"][i]
        })

    caption_records = []

    all_captions = []

    for image_id in validation_ids:

        for caption in validation_captions[image_id]:

            all_captions.append({
                "query_image_id": int(image_id),
                "annotation_id": caption["annotation_id"],
                "caption": caption["caption"]
            })

    for i, query in enumerate(all_captions):

        caption_records.append({
            **query,
            "text_to_image": experiments["text_to_image"][i],
            "text_to_text": experiments["text_to_text"][i]
        })

    with open(
        OUTPUT_DIR / "image_retrieval_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            image_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        OUTPUT_DIR / "caption_retrieval_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            caption_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Summary

    summary = {
        "project": "COCO Image Captioning - RAG",
        "model": MODEL_NAME,
        "device": DEVICE,
        "knowledge_base_images": 10000,
        "validation_images": len(validation_ids),
        "validation_captions": len(all_captions),
        "top_k": TOP_K,
        "experiments": {
            "image_to_image": len(image_records),
            "image_to_text": len(image_records),
            "text_to_image": len(caption_records),
            "text_to_text": len(caption_records)
        },
        "note": (
            "Retrieval rankings and cosine similarity scores are "
            "reported. These results do not constitute semantic "
            "precision or recall without relevance judgments."
        )
    }

    with open(
        OUTPUT_DIR / "validation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Results saved in:", OUTPUT_DIR)


# ============================================================
# 11. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("COCO - VALIDATION RETRIEVAL")
    print("=" * 60)

    print("Model:", MODEL_NAME)
    print("Device:", DEVICE)
    print("Top K:", TOP_K)

    # Load validation data

    (
        kb_ids,
        validation_ids,
        validation_captions
    ) = load_validation_data()

    # Load FAISS indexes

    (
        image_index,
        text_index,
        image_index_ids,
        text_image_ids,
        text_annotation_ids
    ) = load_indexes()

    # Verify index IDs against KB

    if set(image_index_ids.tolist()) != set(kb_ids):
        raise ValueError(
            "Image index IDs do not match Knowledge Base IDs."
        )

    if not set(text_image_ids.tolist()).issubset(set(kb_ids)):
        raise ValueError(
            "Text index contains IDs outside the Knowledge Base."
        )

    # Prepare validation captions

    all_captions = []

    for image_id in validation_ids:

        for caption in validation_captions[image_id]:

            all_captions.append(
                caption["caption"]
            )

    # Load CLIP

    model, processor = load_model()

    # Encode validation queries

    image_embeddings = encode_images(
        model,
        processor,
        validation_ids
    )

    text_embeddings = encode_texts(
        model,
        processor,
        all_captions
    )

    print("\nValidation image embeddings:", image_embeddings.shape)
    print("Validation text embeddings:", text_embeddings.shape)

    # Run retrieval

    experiments = run_experiments(
        image_index,
        text_index,
        image_index_ids,
        text_image_ids,
        text_annotation_ids,
        validation_ids,
        validation_captions,
        image_embeddings,
        text_embeddings
    )

    # Save results

    save_results(
        validation_ids,
        validation_captions,
        experiments
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("VALIDATION RETRIEVAL COMPLETED")
    print("=" * 60)

    print("Validation images:", len(validation_ids))
    print("Validation captions:", len(all_captions))
    print("Image queries:", len(image_embeddings))
    print("Caption queries:", len(text_embeddings))
    print("Execution time: {:.2f} minutes".format(elapsed / 60))

    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
