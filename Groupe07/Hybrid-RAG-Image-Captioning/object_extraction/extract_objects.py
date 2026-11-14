
import json
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "dataset" / "images_train"
SPLITS_PATH = BASE_DIR / "dataset" / "dataset_splits.json"

OUTPUT_DIR = BASE_DIR / "object_extraction" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
CONFIDENCE_THRESHOLD = 0.25

print("Loading dataset splits...")

with open(SPLITS_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)

train_ids = sorted(set(splits["train"]))
validation_ids = sorted(set(splits["validation"]))

print(f"Train images: {len(train_ids)}")
print(f"Validation images: {len(validation_ids)}")

print("\nLoading YOLO model...")
model = YOLO("yolo26n.pt")


def extract_split(image_ids, split_name):

    output_file = OUTPUT_DIR / f"{split_name}_detections.jsonl"

    total_objects = 0
    category_counts = Counter()
    processed_images = 0

    print(f"\nStarting {split_name} extraction...")

    with open(output_file, "w", encoding="utf-8") as output:

        for start in tqdm(
            range(0, len(image_ids), BATCH_SIZE),
            desc=f"Extracting {split_name}"
        ):

            batch_ids = image_ids[start:start + BATCH_SIZE]

            image_paths = [
                IMAGES_DIR / f"{image_id:012d}.jpg"
                for image_id in batch_ids
            ]

            valid_paths = [
                path for path in image_paths if path.exists()
            ]

            if not valid_paths:
                continue

            results = model.predict(
                source=[str(path) for path in valid_paths],
                conf=CONFIDENCE_THRESHOLD,
                batch=BATCH_SIZE,
                verbose=False
            )

            for image_path, result in zip(valid_paths, results):

                detected_objects = []

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    obj = {
                        "label": model.names[class_id],
                        "confidence": round(confidence, 4),
                        "bbox": [
                            round(float(value), 2)
                            for value in box.xyxy[0].tolist()
                        ]
                    }

                    detected_objects.append(obj)
                    category_counts[obj["label"]] += 1

                record = {
                    "image_id": int(image_path.stem),
                    "image": image_path.name,
                    "split": split_name,
                    "number_of_objects": len(detected_objects),
                    "objects": detected_objects
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_objects += len(detected_objects)
                processed_images += 1

    print(f"\n{split_name.upper()} SUMMARY")
    print(f"Images processed: {processed_images}")
    print(f"Total objects detected: {total_objects}")

    if processed_images:
        print(
            f"Average objects per image: "
            f"{total_objects / processed_images:.2f}"
        )

    print("\nTop 10 detected categories:")

    for label, count in category_counts.most_common(10):
        print(f"{label}: {count}")

    print(f"Results saved: {output_file}")


# Extract training objects
extract_split(train_ids, "train")

# Extract validation objects
extract_split(validation_ids, "validation")

print("\n================================")
print("OBJECT EXTRACTION COMPLETED!")
print("================================")
