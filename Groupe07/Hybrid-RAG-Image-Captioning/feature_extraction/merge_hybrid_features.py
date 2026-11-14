
import json
from pathlib import Path

import torch


# ==========================================
# 1. Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLIP_DIR = BASE_DIR / "feature_extraction" / "results"
YOLO_DIR = BASE_DIR / "object_extraction" / "results"

OUTPUT_DIR = BASE_DIR / "hybrid_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load YOLO detections
# ==========================================

def load_detections(split_name):

    detection_file = YOLO_DIR / f"{split_name}_detections.jsonl"

    detections = {}

    with open(detection_file, "r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)
            image_id = int(record["image_id"])

            if image_id in detections:
                raise ValueError(
                    f"Duplicate YOLO image ID: {image_id}"
                )

            detections[image_id] = record

    return detections


# ==========================================
# 3. Merge CLIP and YOLO
# ==========================================

def merge_split(split_name):

    print(f"\nMerging {split_name} features...")

    clip_file = CLIP_DIR / f"{split_name}_clip_embeddings.pt"

    clip_data = torch.load(
        clip_file,
        map_location="cpu",
        weights_only=False
    )

    yolo_data = load_detections(split_name)

    image_ids = clip_data["image_ids"].tolist()
    image_names = clip_data["image_names"]
    embeddings = clip_data["embeddings"]

    if len(image_ids) != len(image_names):
        raise ValueError("CLIP image IDs and names do not match.")

    if len(image_ids) != embeddings.shape[0]:
        raise ValueError("CLIP embeddings count does not match image IDs.")

    missing_ids = [
        image_id for image_id in image_ids
        if image_id not in yolo_data
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} images have no YOLO record. "
            f"First missing IDs: {missing_ids[:10]}"
        )

    # Keep YOLO records in the same order as CLIP embeddings
    object_data = []
    object_texts = []

    for image_id in image_ids:

        record = yolo_data[image_id]
        objects = record["objects"]

        object_data.append(objects)

        # Create a simple textual description from detected labels
        labels = [obj["label"] for obj in objects]
        object_texts.append(", ".join(labels))

    # Final hybrid data
    hybrid_data = {
        "image_ids": clip_data["image_ids"],
        "image_names": image_names,
        "clip_embeddings": embeddings,
        "objects": object_data,
        "object_texts": object_texts,
        "number_of_objects": [
            len(objects) for objects in object_data
        ],
        "embedding_dimension": clip_data["embedding_dimension"],
        "clip_model": clip_data["model"],
        "split": split_name
    }

    output_file = OUTPUT_DIR / f"{split_name}_hybrid_features.pt"

    torch.save(hybrid_data, output_file)

    total_objects = sum(hybrid_data["number_of_objects"])

    print("Images merged:", len(image_ids))
    print("CLIP shape:", tuple(embeddings.shape))
    print("YOLO records:", len(object_data))
    print("Total detected objects:", total_objects)
    print("Output:", output_file)


# ==========================================
# 4. Run both splits
# ==========================================

if __name__ == "__main__":

    merge_split("train")
    merge_split("validation")

    print("\nHYBRID FEATURE MERGING COMPLETED!")

