
import json
from pathlib import Path

from ultralytics import YOLO


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "dataset" / "images_val"
SPLITS_PATH = BASE_DIR / "dataset" / "dataset_splits.json"

OUTPUT_DIR = BASE_DIR / "object_extraction" / "results"
OUTPUT_PATH = OUTPUT_DIR / "test_detections.jsonl"

# YOLO settings
MODEL_NAME = "yolo26n.pt"
BATCH_SIZE = 16
CONFIDENCE_THRESHOLD = 0.25


def main():
    print("=" * 60)
    print("YOLO OBJECT EXTRACTION - TEST")
    print("=" * 60)

    # Load Test image IDs
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = splits["test"]

    print(f"Test images: {len(test_ids)}")

    # Build image paths
    image_paths = []

    for image_id in test_ids:
        image_name = f"{int(image_id):012d}.jpg"
        image_path = IMAGES_DIR / image_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing Test image: {image_path}"
            )

        image_paths.append((int(image_id), image_name, image_path))

    print("All Test images were found.")

    # Load YOLO model
    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_objects = 0
    processed_images = 0

    # Extract objects in batches
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:

        for start in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[start:start + BATCH_SIZE]

            paths = [str(item[2]) for item in batch]

            results = model.predict(
                source=paths,
                conf=CONFIDENCE_THRESHOLD,
                batch=BATCH_SIZE,
                verbose=False
            )

            for item, result in zip(batch, results):
                image_id, image_name, _ = item
                detected_objects = []

                if result.boxes is not None:
                    boxes = result.boxes

                    for cls_id, confidence, bbox in zip(
                        boxes.cls.cpu().tolist(),
                        boxes.conf.cpu().tolist(),
                        boxes.xyxy.cpu().tolist()
                    ):
                        label = model.names[int(cls_id)]

                        detected_objects.append({
                            "label": label,
                            "confidence": round(float(confidence), 4),
                            "bbox": [
                                round(float(value), 2)
                                for value in bbox
                            ]
                        })

                record = {
                    "image_id": image_id,
                    "image": image_name,
                    "split": "test",
                    "number_of_objects": len(detected_objects),
                    "objects": detected_objects
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_objects += len(detected_objects)
                processed_images += 1

            print(
                f"Processed: {processed_images}/{len(image_paths)}"
            )

    average_objects = (
        total_objects / processed_images
        if processed_images else 0
    )

    print("\n" + "=" * 60)
    print("TEST EXTRACTION COMPLETED")
    print("=" * 60)
    print(f"Images processed: {processed_images}")
    print(f"Total detected objects: {total_objects}")
    print(f"Average objects per image: {average_objects:.2f}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
