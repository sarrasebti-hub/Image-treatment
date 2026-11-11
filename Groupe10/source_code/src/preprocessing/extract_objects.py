import json
from pathlib import Path
from ultralytics import YOLO

DATA_DIR = Path("data/processed")
SPLITS = ["knowledge_base", "validation", "test"]

CONFIDENCE_THRESHOLD = 0.25
SAVE_EVERY = 100

model = YOLO("yolov8n.pt")


def save_metadata(metadata, path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)


def extract_objects(split):
    split_dir = DATA_DIR / split
    images_dir = split_dir / "images"
    metadata_path = split_dir / "metadata.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    processed_now = 0
    total_objects = 0

    remaining = sum(1 for item in metadata if "objects" not in item)

    print(f"\n===== {split.upper()} =====")
    print(f"Already processed: {len(metadata) - remaining}/{len(metadata)}")
    print(f"Remaining: {remaining}")

    if remaining == 0:
        print("All images already processed.")
        return

    for item in metadata:
        if "objects" in item:
            continue

        image_path = images_dir / item["file_name"]

        try:
            results = model.predict(
                source=str(image_path),
                conf=CONFIDENCE_THRESHOLD,
                device="cpu",
                verbose=False
            )

            objects = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])

                    objects.append({
                        "name": model.names[class_id],
                        "confidence": round(float(box.conf[0]), 3),
                        "bbox": [
                            round(float(x), 2)
                            for x in box.xyxy[0].tolist()
                        ]
                    })

            item["objects"] = objects
            item["num_objects"] = len(objects)

            processed_now += 1
            total_objects += len(objects)

            if processed_now % 10 == 0:
                print(
                    f"{split}: {processed_now}/{remaining} "
                    "new images processed"
                )

            if processed_now % SAVE_EVERY == 0:
                save_metadata(metadata, metadata_path)
                print("Progress saved.")

        except Exception as error:
            print(f"Error processing {item['file_name']}: {error}")

    save_metadata(metadata, metadata_path)

    total_processed = sum(
        1 for item in metadata if "objects" in item
    )

    print(f"\nCompleted {split}:")
    print(f"New images processed: {processed_now}")
    print(f"Total images processed: {total_processed}/{len(metadata)}")
    print(f"Objects detected in this run: {total_objects}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    for split in SPLITS:
        extract_objects(split)

    print("\nAll splits finished!")
