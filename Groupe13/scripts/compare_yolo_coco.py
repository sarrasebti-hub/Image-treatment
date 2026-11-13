
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

YOLO_FILE = (
    ROOT / "data" / "visual_detections_test"
    / "yolo_test_results.json"
)

COCO_FILE = ROOT / "data" / "metadata" / "kb_metadata.json"

OUTPUT_DIR = ROOT / "data" / "visual_detections_test"
OUTPUT_JSON = OUTPUT_DIR / "yolo_coco_comparison.json"
OUTPUT_CSV = OUTPUT_DIR / "yolo_coco_comparison.csv"

IOU_THRESHOLD = 0.50


def calculate_iou(box1, box2):
    """
    box1 et box2 au format [x1, y1, x2, y2].
    """

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )
    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def coco_bbox_to_xyxy(bbox):
    """Convertir [x, y, largeur, hauteur] en [x1,y1,x2,y2]."""

    x, y, width, height = bbox

    return [
        x,
        y,
        x + width,
        y + height
    ]


# Charger les données
with open(YOLO_FILE, "r", encoding="utf-8") as f:
    yolo_results = json.load(f)

with open(COCO_FILE, "r", encoding="utf-8") as f:
    coco_metadata = json.load(f)

# Indexer les métadonnées COCO par image_id
coco_by_id = {
    image["image_id"]: image
    for image in coco_metadata
}

comparison_results = []

total_tp = 0
total_fp = 0
total_fn = 0

for yolo_image in yolo_results:

    image_id = yolo_image["image_id"]

    if image_id not in coco_by_id:
        print("Image absente de COCO :", image_id)
        continue

    coco_image = coco_by_id[image_id]

    # Préparer les objets COCO
    coco_objects = []

    for obj in coco_image["objects"]:
        coco_objects.append({
            "category": obj["category"],
            "bbox": coco_bbox_to_xyxy(obj["bbox"])
        })

    # Préparer les détections YOLO
    yolo_objects = []

    for obj in yolo_image["objects"]:
        yolo_objects.append({
            "category": obj["class_name"],
            "bbox": obj["bbox_xyxy"],
            "confidence": obj["confidence"]
        })

    # Créer les correspondances candidates
    candidates = []

    for yi, yolo_obj in enumerate(yolo_objects):
        for ci, coco_obj in enumerate(coco_objects):

            # Les catégories doivent être identiques
            if yolo_obj["category"] != coco_obj["category"]:
                continue

            iou = calculate_iou(
                yolo_obj["bbox"],
                coco_obj["bbox"]
            )

            if iou >= IOU_THRESHOLD:
                candidates.append((iou, yi, ci))

    # Trier par IoU décroissant
    candidates.sort(reverse=True)

    matched_yolo = set()
    matched_coco = set()
    matches = []

    # Correspondance un-à-un
    for iou, yi, ci in candidates:

        if yi in matched_yolo or ci in matched_coco:
            continue

        matched_yolo.add(yi)
        matched_coco.add(ci)

        matches.append({
            "category": yolo_objects[yi]["category"],
            "iou": round(iou, 4),
            "confidence": yolo_objects[yi]["confidence"]
        })

    tp = len(matches)
    fp = len(yolo_objects) - tp
    fn = len(coco_objects) - tp

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    comparison_results.append({
        "image_id": image_id,
        "file_name": yolo_image["image_file"],
        "coco_objects": len(coco_objects),
        "yolo_detections": len(yolo_objects),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "matches": matches
    })

# Calculer les métriques globales
precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

summary = {
    "images_compared": len(comparison_results),
    "iou_threshold": IOU_THRESHOLD,
    "total_tp": total_tp,
    "total_fp": total_fp,
    "total_fn": total_fn,
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "images": comparison_results
}

# Sauvegarder JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

# Sauvegarder CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")

    writer.writerow([
        "image_id",
        "file_name",
        "coco_objects",
        "yolo_detections",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall"
    ])

    for image in comparison_results:
        writer.writerow([
            image["image_id"],
            image["file_name"],
            image["coco_objects"],
            image["yolo_detections"],
            image["true_positives"],
            image["false_positives"],
            image["false_negatives"],
            image["precision"],
            image["recall"]
        ])

# Afficher les résultats
print("\n========== COMPARAISON YOLO / COCO ==========")

print("Images comparées :", len(comparison_results))
print("Seuil IoU :", IOU_THRESHOLD)

print("\n--- Résultats globaux ---")
print("True Positives (TP) :", total_tp)
print("False Positives (FP) :", total_fp)
print("False Negatives (FN) :", total_fn)

print("\n--- Métriques ---")
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1-score  :", round(f1, 4))

print("\n--- Résultats par image ---")

for image in comparison_results:
    print(
        image["file_name"],
        "| TP:", image["true_positives"],
        "| FP:", image["false_positives"],
        "| FN:", image["false_negatives"]
    )

print("\nFichiers générés :")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
