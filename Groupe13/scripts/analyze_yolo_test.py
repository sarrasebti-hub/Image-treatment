
import json
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "visual_detections_test"
    / "yolo_test_results.json"
)

OUTPUT_DIR = ROOT / "data" / "visual_detections_test"
OUTPUT_JSON = OUTPUT_DIR / "yolo_test_statistics.json"
OUTPUT_CSV = OUTPUT_DIR / "yolo_test_class_statistics.csv"

# Charger les résultats YOLO
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

class_confidences = defaultdict(list)
image_statistics = []

total_detections = 0
all_confidences = []

for image in results:
    image_id = image["image_id"]
    detections = image["objects"]

    total_detections += len(detections)

    image_statistics.append({
        "image_id": image_id,
        "image_file": image["image_file"],
        "number_of_objects": len(detections)
    })

    for obj in detections:
        class_name = obj["class_name"]
        confidence = obj["confidence"]

        class_confidences[class_name].append(confidence)
        all_confidences.append(confidence)

# Statistiques globales
class_statistics = {}

for class_name, scores in sorted(class_confidences.items()):
    class_statistics[class_name] = {
        "count": len(scores),
        "mean_confidence": round(mean(scores), 4),
        "min_confidence": round(min(scores), 4),
        "max_confidence": round(max(scores), 4)
    }

summary = {
    "images_processed": len(results),
    "total_detections": total_detections,
    "number_of_classes": len(class_statistics),
    "global_mean_confidence": round(mean(all_confidences), 4)
        if all_confidences else None,
    "classes": class_statistics,
    "images": image_statistics
}

# Sauvegarder le JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

# Sauvegarder les statistiques par classe en CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([
        "class_name",
        "count",
        "mean_confidence",
        "min_confidence",
        "max_confidence"
    ])

    for class_name, stats in class_statistics.items():
        writer.writerow([
            class_name,
            stats["count"],
            stats["mean_confidence"],
            stats["min_confidence"],
            stats["max_confidence"]
        ])

# Afficher les résultats
print("\n========== STATISTIQUES YOLO ==========")
print("Images traitées :", len(results))
print("Total des détections :", total_detections)
print("Nombre de catégories :", len(class_statistics))
print("Confiance moyenne globale :", summary["global_mean_confidence"])

print("\n--- Statistiques par catégorie ---")

for class_name, stats in class_statistics.items():
    print(
        f"{class_name}: "
        f"{stats['count']} objets | "
        f"moyenne={stats['mean_confidence']} | "
        f"min={stats['min_confidence']} | "
        f"max={stats['max_confidence']}"
    )

print("\n--- Détections par image ---")

for image in image_statistics:
    print(image["image_file"], ":", image["number_of_objects"])

print("\nFichiers sauvegardés :")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
