
import json
import random
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

SPLIT_FILE = ROOT / "data/splits/knowledge_base_image_ids.json"
IMAGE_DIR = ROOT / "data/train2017"
COCO_FILE = ROOT / "data/metadata/kb_metadata.json"

OUTPUT_DIR = ROOT / "data/visual_detections_test"
OUTPUT_FILE = OUTPUT_DIR / "threshold_experiments.json"

THRESHOLDS = [0.25, 0.40, 0.50]
IOU_THRESHOLD = 0.50

# Sélectionner exactement les mêmes 100 images que le pilote
with open(SPLIT_FILE, encoding="utf-8") as f:
    random.seed(42)
    selected_ids = random.sample(json.load(f), 100)

with open(COCO_FILE, encoding="utf-8") as f:
    coco_data = json.load(f)

coco_by_id = {x["image_id"]: x for x in coco_data}


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


model = YOLO("yolo26n.pt")
experiments = []

for threshold in THRESHOLDS:

    total_tp = total_fp = total_fn = 0
    total_detections = 0

    for image_id in selected_ids:

        image_path = IMAGE_DIR / f"{image_id:012d}.jpg"

        prediction = model.predict(
            source=str(image_path),
            conf=threshold,
            device="cpu",
            verbose=False
        )[0]

        yolo_objects = []

        for box in prediction.boxes:
            class_id = int(box.cls.item())
            yolo_objects.append({
                "category": model.names[class_id],
                "bbox": box.xyxy[0].tolist()
            })

        # Objets COCO : exclure les annotations iscrowd
        coco_objects = [
            {
                "category": obj["category"],
                "bbox": [
                    obj["bbox"][0],
                    obj["bbox"][1],
                    obj["bbox"][0] + obj["bbox"][2],
                    obj["bbox"][1] + obj["bbox"][3]
                ]
            }
            for obj in coco_by_id[image_id]["objects"]
            if obj.get("iscrowd", 0) == 0
        ]

        candidates = []

        for yi, yolo in enumerate(yolo_objects):
            for ci, coco in enumerate(coco_objects):

                if yolo["category"] != coco["category"]:
                    continue

                score = iou(yolo["bbox"], coco["bbox"])

                if score >= IOU_THRESHOLD:
                    candidates.append((score, yi, ci))

        candidates.sort(reverse=True)

        used_yolo = set()
        used_coco = set()

        for score, yi, ci in candidates:
            if yi not in used_yolo and ci not in used_coco:
                used_yolo.add(yi)
                used_coco.add(ci)

        tp = len(used_yolo)
        fp = len(yolo_objects) - tp
        fn = len(coco_objects) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_detections += len(yolo_objects)

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

    experiments.append({
        "confidence_threshold": threshold,
        "images": len(selected_ids),
        "detections": total_detections,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    })

    print(
        f"\nThreshold: {threshold}"
        f"\nDetections: {total_detections}"
        f"\nTP: {total_tp} | FP: {total_fp} | FN: {total_fn}"
        f"\nPrecision: {precision:.4f}"
        f"\nRecall: {recall:.4f}"
        f"\nF1-score: {f1:.4f}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(experiments, f, indent=4)

print("\nRésultats sauvegardés :", OUTPUT_FILE)
