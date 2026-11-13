
import json
import csv
import random
import time
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO


# =========================
# CONFIGURATION
# =========================

PROJECT_DIR = Path(__file__).resolve().parent.parent

KB_IDS_FILE = PROJECT_DIR / "data" / "splits" / "knowledge_base_image_ids.json"
IMAGES_DIR = PROJECT_DIR / "data" / "train2017"
OUTPUT_DIR = PROJECT_DIR / "data" / "visual_detections" / "pilot_100"

MODEL_PATH = PROJECT_DIR / "yolo26n.pt"

NUM_IMAGES = 100
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cpu"
NUM_ANNOTATED_IMAGES = 10


def main():
    print("========== YOLO PILOT: 100 IMAGES ==========")

    # 1. Verify input files
    if not KB_IDS_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {KB_IDS_FILE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    with open(KB_IDS_FILE, "r", encoding="utf-8") as f:
        kb_ids = json.load(f)

    if len(kb_ids) != 10000:
        raise ValueError(
            f"La Knowledge Base doit contenir 10000 images, "
            f"mais contient {len(kb_ids)}."
        )

    if len(set(kb_ids)) != len(kb_ids):
        raise ValueError("La Knowledge Base contient des identifiants dupliqués.")

    if len(kb_ids) < NUM_IMAGES:
        raise ValueError("La Knowledge Base contient moins de 100 images.")

    # 2. Select 100 reproducible images
    random.seed(RANDOM_SEED)
    selected_ids = random.sample(kb_ids, NUM_IMAGES)

    image_paths = []

    for image_id in selected_ids:
        image_path = IMAGES_DIR / f"{int(image_id):012d}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        image_paths.append((int(image_id), image_path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotated_dir = OUTPUT_DIR / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images sélectionnées : {len(image_paths)}")
    print(f"Seuil de confiance : {CONFIDENCE_THRESHOLD}")
    print(f"Modèle : {MODEL_PATH.name}")
    print(f"Appareil : {DEVICE}")

    # 3. Load model
    print("\nChargement du modèle YOLO...")
    model = YOLO(str(MODEL_PATH))

    all_images = []
    csv_rows = []
    category_stats = defaultdict(list)

    total_detections = 0
    zero_detection_images = 0
    confidence_values = []

    # 4. Run detection
    start_time = time.time()

    for index, (image_id, image_path) in enumerate(image_paths, start=1):

        result = model.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            device=DEVICE,
            verbose=False
        )[0]

        image_detections = []

        if result.boxes is not None:
            for box in result.boxes:

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = [
                    float(value) for value in box.xyxy[0].tolist()
                ]

                class_name = result.names[class_id]

                detection = {
                    "class_id": class_id,
                    "category": class_name,
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                }

                image_detections.append(detection)
                confidence_values.append(confidence)
                category_stats[class_name].append(confidence)

                csv_rows.append({
                    "image_id": image_id,
                    "category": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 6),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

        detection_count = len(image_detections)
        total_detections += detection_count

        if detection_count == 0:
            zero_detection_images += 1

        all_images.append({
            "image_id": image_id,
            "file_name": image_path.name,
            "detection_count": detection_count,
            "detections": image_detections
        })

        # Save annotated versions of the first 10 images
        if index <= NUM_ANNOTATED_IMAGES:
            annotated_path = annotated_dir / f"{image_id:012d}_detected.jpg"
            result.save(filename=str(annotated_path))

        if index % 10 == 0:
            print(
                f"Progression : {index}/{NUM_IMAGES} images "
                f"| Détections cumulées : {total_detections}"
            )

    elapsed_time = time.time() - start_time

    # 5. Calculate statistics
    average_detections = total_detections / NUM_IMAGES

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0
    )

    category_summary = {}

    for category, scores in sorted(category_stats.items()):
        category_summary[category] = {
            "count": len(scores),
            "mean_confidence": round(sum(scores) / len(scores), 4),
            "min_confidence": round(min(scores), 4),
            "max_confidence": round(max(scores), 4)
        }

    confidence_distribution = {
        "0.25_to_0.40": sum(0.25 <= x < 0.40 for x in confidence_values),
        "0.40_to_0.60": sum(0.40 <= x < 0.60 for x in confidence_values),
        "0.60_to_0.80": sum(0.60 <= x < 0.80 for x in confidence_values),
        "0.80_to_1.00": sum(0.80 <= x <= 1.00 for x in confidence_values)
    }

    statistics = {
        "experiment": "YOLO pilot on 100 Knowledge Base images",
        "model": MODEL_PATH.name,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "device": DEVICE,
        "random_seed": RANDOM_SEED,
        "images_processed": NUM_IMAGES,
        "images_with_no_detections": zero_detection_images,
        "total_detections": total_detections,
        "average_detections_per_image": round(average_detections, 3),
        "number_of_categories": len(category_summary),
        "mean_confidence": round(average_confidence, 4),
        "confidence_distribution": confidence_distribution,
        "processing_time_seconds": round(elapsed_time, 2),
        "average_time_per_image_seconds": round(elapsed_time / NUM_IMAGES, 3),
        "images_per_second": round(NUM_IMAGES / elapsed_time, 3),
        "categories": category_summary
    }

    # 6. Save JSON results
    results_path = OUTPUT_DIR / "pilot_100_detections.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "configuration": {
                "model": MODEL_PATH.name,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "device": DEVICE,
                "random_seed": RANDOM_SEED
            },
            "images": all_images
        }, f, ensure_ascii=False, indent=2)

    # 7. Save CSV results
    csv_path = OUTPUT_DIR / "pilot_100_detections.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image_id", "category", "class_id", "confidence",
            "x1", "y1", "x2", "y2"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # 8. Save statistics
    stats_path = OUTPUT_DIR / "pilot_100_statistics.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    # 9. Print summary
    print("\n========== PILOT SUMMARY ==========")
    print(f"Images traitées : {NUM_IMAGES}")
    print(f"Total détections : {total_detections}")
    print(f"Moyenne par image : {average_detections:.2f}")
    print(f"Images sans détection : {zero_detection_images}")
    print(f"Catégories détectées : {len(category_summary)}")
    print(f"Confiance moyenne : {average_confidence:.4f}")
    print(f"Temps total : {elapsed_time:.2f} secondes")
    print(f"Temps moyen/image : {elapsed_time / NUM_IMAGES:.3f} secondes")

    print("\nDétections par catégorie:")

    for category, stats in category_summary.items():
        print(
            f"- {category}: {stats['count']} "
            f"(confiance moyenne: {stats['mean_confidence']})"
        )

    print("\nFichiers sauvegardés dans :", OUTPUT_DIR)
    print("- pilot_100_detections.json")
    print("- pilot_100_detections.csv")
    print("- pilot_100_statistics.json")
    print("- annotated_images/ (10 images)")


if __name__ == "__main__":
    main()
