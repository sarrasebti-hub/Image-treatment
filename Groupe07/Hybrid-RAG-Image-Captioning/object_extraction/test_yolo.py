
from pathlib import Path
from ultralytics import YOLO
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "dataset" / "images_train"
OUTPUT_DIR = BASE_DIR / "object_extraction" / "test_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load pretrained YOLO model
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# Select 10 images
image_paths = sorted(IMAGES_DIR.glob("*.jpg"))[:10]

results_data = []

print(f"\nProcessing {len(image_paths)} images...")

for image_path in image_paths:

    print(f"\nProcessing: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    result = results[0]
    detected_objects = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detected_objects.append({
            "label": model.names[class_id],
            "confidence": round(confidence, 4),
            "bbox": [
                round(float(x), 2)
                for x in box.xyxy[0].tolist()
            ]
        })

    results_data.append({
        "image": image_path.name,
        "objects": detected_objects,
        "number_of_objects": len(detected_objects)
    })

    # Save annotated image
    annotated_image = result.plot()

    output_path = OUTPUT_DIR / f"detected_{image_path.stem}.jpg"

    from PIL import Image
    Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

    print(f"Detected objects: {len(detected_objects)}")

# Save detection results
json_path = OUTPUT_DIR / "test_detections.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=4, ensure_ascii=False)

total_objects = sum(
    item["number_of_objects"] for item in results_data
)

print("\n========== TEST SUMMARY ==========")
print(f"Images processed: {len(results_data)}")
print(f"Total detected objects: {total_objects}")
print(f"Average objects per image: {total_objects / len(results_data):.2f}")
print(f"Results saved in: {OUTPUT_DIR}")
