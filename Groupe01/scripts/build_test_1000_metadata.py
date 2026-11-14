from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_common import EXPERIMENT_ROOT, read_json, write_json


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    raw = ROOT / "data" / "raw" / "annotations"
    split = read_json(EXPERIMENT_ROOT / "splits" / "test_1000.json")
    instances = read_json(raw / "instances_val2017.json")
    captions = read_json(raw / "captions_val2017.json")
    categories = {int(row["id"]): row for row in instances["categories"]}
    coco_images = {int(row["id"]): row for row in instances["images"]}
    selected = {int(row["image_id"]) for row in split["images"]}

    objects_by_image: dict[int, list[dict]] = {}
    for annotation in instances["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in selected or annotation.get("iscrowd", 0):
            continue
        category = categories[int(annotation["category_id"])]
        objects_by_image.setdefault(image_id, []).append({
            "category_id": int(annotation["category_id"]),
            "category": category["name"],
            "supercategory": category["supercategory"],
            "bbox": annotation["bbox"],
        })

    captions_by_image: dict[int, list[str]] = {}
    caption_ids_by_image: dict[int, list[int]] = {}
    for annotation in captions["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in selected:
            captions_by_image.setdefault(image_id, []).append(annotation["caption"])
            caption_ids_by_image.setdefault(image_id, []).append(int(annotation["id"]))

    reference_rows = []
    metadata_rows = []
    for row in split["images"]:
        image_id = int(row["image_id"])
        ordered = sorted(zip(caption_ids_by_image[image_id], captions_by_image[image_id]), key=lambda pair: pair[0])
        for caption_id, caption in ordered[:5]:
            reference_rows.append({
                "caption_id": caption_id,
                "image_id": image_id,
                "caption_raw": caption,
            })

        objects = objects_by_image[image_id]
        counts = Counter(obj["category"] for obj in objects)
        metadata_rows.append({
            "image_id": image_id,
            "split": "test_1000",
            "n_objects": len(objects),
            "category_counts": dict(counts),
            "n_categories": len(counts),
            "supercategories": sorted({obj["supercategory"] for obj in objects}),
            "objects": objects,
            "metadata_sources": {"objects": "coco_gt_annotations"},
        })

    write_jsonl(EXPERIMENT_ROOT / "tables" / "eval_refs_test_1000.jsonl", reference_rows)
    write_jsonl(EXPERIMENT_ROOT / "tables" / "metadata_test_1000.jsonl", metadata_rows)
    write_json(EXPERIMENT_ROOT / "manifests" / "preparation_manifest.json", {
        "coco_images_file": str(raw / "instances_val2017.json"),
        "caption_file": str(raw / "captions_val2017.json"),
        "reference_count": len(reference_rows),
        "query_count": len(metadata_rows),
    })
    print(f"Created references for {len(metadata_rows)} images and {len(reference_rows)} captions")


if __name__ == "__main__":
    main()
