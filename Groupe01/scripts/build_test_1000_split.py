from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_common import EXPERIMENT_ROOT, deterministic_sample, read_json, sha256_ids, split_ids, write_json


ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_annotations(raw: Path) -> tuple[Path, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "annotations_trainval2017.zip"
    annotations = raw / "annotations"
    if not (annotations / "instances_val2017.json").exists():
        if not archive.exists() or archive.stat().st_size == 0:
            print(f"Downloading COCO annotations to {archive}")
            urllib.request.urlretrieve(ANN_URL, archive)
        if archive.stat().st_size < 100_000_000:
            raise ValueError(f"Incomplete COCO annotations archive: {archive}")
        annotations.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            wanted = ("instances_val2017.json", "captions_val2017.json")
            for name in wanted:
                member = f"annotations/{name}"
                zipped.extract(member, raw)
    return annotations / "instances_val2017.json", annotations / "captions_val2017.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    original_test_path = ROOT / "artifacts" / "splits" / "test.json"
    original_ids = split_ids(original_test_path)
    if len(original_ids) != 100:
        raise ValueError(f"Expected 100 original test images, found {len(original_ids)}")

    instances_path, captions_path = download_annotations(args.raw)
    instances = read_json(instances_path)
    captions = read_json(captions_path)

    caption_counts: dict[int, int] = {}
    for row in captions["annotations"]:
        image_id = int(row["image_id"])
        caption_counts[image_id] = caption_counts.get(image_id, 0) + 1

    object_counts: dict[int, int] = {}
    for row in instances["annotations"]:
        if not row.get("iscrowd", 0):
            image_id = int(row["image_id"])
            object_counts[image_id] = object_counts.get(image_id, 0) + 1

    candidates = []
    for image in instances["images"]:
        image_id = int(image["id"])
        if image_id in original_ids:
            continue
        if caption_counts.get(image_id, 0) < 5:
            continue
        if object_counts.get(image_id, 0) < 1:
            continue
        if min(int(image["width"]), int(image["height"])) < 200:
            continue
        candidates.append((image_id, image))

    if len(candidates) < args.count:
        raise ValueError(f"Only {len(candidates)} eligible disjoint val2017 images are available")

    selected = deterministic_sample(
        {image_id for image_id, _ in candidates},
        args.count,
        args.seed,
    )
    by_id = {image_id: image for image_id, image in candidates}
    records = []
    for image_id in selected:
        image = by_id[image_id]
        file_name = image["file_name"]
        records.append({
            "image_id": image_id,
            "source": "val2017",
            "file_name": file_name,
            "coco_url": f"http://images.cocodataset.org/val2017/{file_name}",
            "width": int(image["width"]),
            "height": int(image["height"]),
            "n_objects": object_counts[image_id],
            "n_captions": caption_counts[image_id],
        })

    split_path = EXPERIMENT_ROOT / "splits" / "test_1000.json"
    write_json(split_path, {
        "split": "test_1000",
        "version": "test-1000-val2017-v1",
        "seed": args.seed,
        "source": "val2017",
        "images": records,
    })

    manifest = {
        "experiment": "test_1000_val2017",
        "source": "val2017",
        "seed": args.seed,
        "count": len(records),
        "original_test_count": len(original_ids),
        "original_test_id_hash": sha256_ids(original_ids),
        "selected_id_hash": sha256_ids(selected),
        "overlap_count": len(set(selected) & original_ids),
        "eligible_candidate_count": len(candidates),
    }
    write_json(EXPERIMENT_ROOT / "manifests" / "split_manifest.json", manifest)
    print(f"Created {split_path} with {len(records)} images")


if __name__ == "__main__":
    main()
