
import json
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent.parent

CLIP_PATH = (
    BASE_DIR / "feature_extraction" / "results"
    / "test_clip_embeddings.pt"
)

YOLO_PATH = (
    BASE_DIR / "object_extraction" / "results"
    / "test_detections.jsonl"
)

OUTPUT_DIR = BASE_DIR / "hybrid_features"
OUTPUT_PATH = OUTPUT_DIR / "test_hybrid_features.pt"


def main():
    print("=" * 60)
    print("TEST HYBRID FEATURE MERGING")
    print("=" * 60)

    # Load CLIP features
    print("\nLoading CLIP embeddings...")

    clip_data = torch.load(
        CLIP_PATH,
        map_location="cpu",
        weights_only=False
    )

    clip_ids = clip_data["image_ids"].tolist()
    clip_names = clip_data["image_names"]
    clip_embeddings = clip_data["embeddings"]

    print(f"CLIP images: {len(clip_ids)}")
    print(f"CLIP shape: {tuple(clip_embeddings.shape)}")

    # Load YOLO detections
    print("\nLoading YOLO detections...")

    yolo_records = {}

    with open(YOLO_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            image_id = int(record["image_id"])

            if image_id in yolo_records:
                raise ValueError(f"Duplicate YOLO ID: {image_id}")

            yolo_records[image_id] = record

    print(f"YOLO images: {len(yolo_records)}")

    # Verify alignment
    if len(clip_ids) != len(yolo_records):
        raise ValueError("CLIP and YOLO image counts do not match.")

    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Duplicate IDs found in CLIP data.")

    missing_ids = set(clip_ids) - set(yolo_records)

    if missing_ids:
        raise ValueError(
            f"Missing YOLO records for IDs: {list(missing_ids)[:10]}"
        )

    print("CLIP and YOLO IDs are aligned.")

    # Arrange YOLO information in CLIP order
    objects = []
    object_texts = []
    number_of_objects = []

    for image_id in clip_ids:
        record = yolo_records[image_id]
        detected_objects = record["objects"]

        objects.append(detected_objects)

        labels = [
            obj["label"] for obj in detected_objects
        ]

        object_texts.append(", ".join(labels))
        number_of_objects.append(len(detected_objects))

    # Create hybrid dataset
    hybrid_data = {
        "image_ids": torch.tensor(clip_ids, dtype=torch.int64),
        "image_names": clip_names,
        "clip_embeddings": clip_embeddings,
        "objects": objects,
        "object_texts": object_texts,
        "number_of_objects": number_of_objects,
        "embedding_dimension": clip_embeddings.shape[1],
        "model": clip_data["model"],
        "pretrained": clip_data["pretrained"],
        "split": "test"
    }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(hybrid_data, OUTPUT_PATH)

    total_objects = sum(number_of_objects)

    print("\n" + "=" * 60)
    print("TEST HYBRID MERGING COMPLETED")
    print("=" * 60)
    print(f"Images: {len(clip_ids)}")
    print(f"CLIP shape: {tuple(clip_embeddings.shape)}")
    print(f"Total YOLO objects: {total_objects}")
    print(f"Average objects/image: {total_objects / len(clip_ids):.2f}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
