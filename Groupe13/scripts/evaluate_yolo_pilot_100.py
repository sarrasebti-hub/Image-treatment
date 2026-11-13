
import json
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent

PILOT_FILE = (
    PROJECT_DIR / "data" / "visual_detections" /
    "pilot_100" / "pilot_100_detections.json"
)
METADATA_FILE = PROJECT_DIR / "data" / "metadata" / "kb_metadata.json"

OUTPUT_DIR = PROJECT_DIR / "data" / "visual_detections" / "pilot_100"
IOU_THRESHOLD = 0.50


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def main():
    with open(PILOT_FILE, encoding="utf-8") as f:
        pilot = json.load(f)

    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_by_id = {
        int(item["image_id"]): item
        for item in metadata
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_image = []
    category_counts = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0}
    )

    for image in pilot["images"]:
        image_id = int(image["image_id"])

        if image_id not in metadata_by_id:
            raise ValueError(
                f"Image {image_id} absente de kb_metadata.json"
            )

        # YOLO predictions: xyxy
        predictions = image["detections"]

        # COCO ground truth: xywh -> xyxy
        ground_truth = []

        for obj in metadata_by_id[image_id]["objects"]:
            if obj.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = obj["bbox"]

            ground_truth.append({
                "category": obj["category"],
                "bbox": [x, y, x + w, y + h]
            })

        # Build candidate matches
        candidates = []

        for pred_index, pred in enumerate(predictions):
            for gt_index, gt in enumerate(ground_truth):
                if pred["category"] != gt["category"]:
                    continue

                iou = calculate_iou(
                    pred["bbox_xyxy"],
                    gt["bbox"]
                )

                if iou >= IOU_THRESHOLD:
                    candidates.append(
                        (iou, pred_index, gt_index)
                    )

        # Greedy one-to-one matching, highest IoU first
        candidates.sort(reverse=True)

        matched_predictions = set()
        matched_ground_truth = set()

        for iou, pred_index, gt_index in candidates:
            if (
                pred_index not in matched_predictions
                and gt_index not in matched_ground_truth
            ):
                matched_predictions.add(pred_index)
                matched_ground_truth.add(gt_index)

        tp = len(matched_predictions)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-category statistics
        for index, pred in enumerate(predictions):
            category = pred["category"]
            if index in matched_predictions:
                category_counts[category]["TP"] += 1
            else:
                category_counts[category]["FP"] += 1

        for index, gt in enumerate(ground_truth):
            category = gt["category"]
            if index not in matched_ground_truth:
                category_counts[category]["FN"] += 1

        per_image.append({
            "image_id": image_id,
            "yolo_detections": len(predictions),
            "coco_objects": len(ground_truth),
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

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
        "images_compared": len(per_image),
        "iou_threshold": IOU_THRESHOLD,
        "matching_rule": "Same category and IoU >= 0.50",
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "per_image": per_image,
        "per_category": dict(category_counts)
    }

    # Save JSON
    json_path = OUTPUT_DIR / "pilot_100_evaluation.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save per-image CSV
    csv_path = OUTPUT_DIR / "pilot_100_evaluation.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id", "yolo_detections",
                "coco_objects", "TP", "FP", "FN"
            ]
        )
        writer.writeheader()
        writer.writerows(per_image)

    print("\n========== YOLO / COCO EVALUATION ==========")
    print(f"Images compared : {len(per_image)}")
    print(f"IoU threshold   : {IOU_THRESHOLD}")
    print(f"TP              : {total_tp}")
    print(f"FP              : {total_fp}")
    print(f"FN              : {total_fn}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1-score        : {f1:.4f}")

    print("\nFichiers sauvegardés :")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
