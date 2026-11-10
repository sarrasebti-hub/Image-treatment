"""Simplify the three existing per-split image metadata files in place."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"train": 10_000, "validation": 100, "test": 1_000}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _hashes(root: Path) -> dict[str, str]:
    """Hash the frozen manifests, captions, embeddings, and FAISS artifacts."""
    protected = []
    for folder, patterns in (
        ("data/selected", ("*_ids.json", "*_ids.csv")),
        ("data/captions", ("train.json", "train.csv", "validation.json", "validation.csv", "test.json", "test.csv")),
        ("embeddings", ("*.npy", "*.metadata.json", "*.run.json")),
        ("vector_database", ("*.index", "*.metadata.json")),
    ):
        base = root / folder
        for pattern in patterns:
            protected.extend(path for path in base.rglob(pattern) if path.is_file())
    result = {}
    for path in sorted(set(protected)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


def _simplify(split: str, records: list[dict]) -> list[dict]:
    simplified = []
    for row in records:
        ident = row["image_identification"]
        overview = row["scene_overview"]
        objects = []
        for obj in row["object_annotations"]:
            box = obj["bounding_box"]
            objects.append({
                "name": obj["category_name"],
                "bbox": [box[key] for key in ("x", "y", "width", "height")],
            })
        categories = list(dict.fromkeys(obj["name"] for obj in objects))
        simplified.append({
            "image_id": row["image_id"],
            "filename": ident["original_filename"],
            "split": split,
            "width": ident["width_pixels"],
            "height": ident["height_pixels"],
            "objects": objects,
            "object_count": len(objects),
            "categories": categories,
        })
    return simplified


def _validate(root: Path, manifests: dict[str, list[int]]) -> dict:
    summary = {}
    for split in SPLITS:
        path = root / "data" / "metadata" / f"{split}.json"
        rows = _read(path)
        expected_fields = {"image_id", "filename", "split", "width", "height", "objects", "object_count", "categories"}
        expected_ids = manifests[split]
        errors = []
        if len(rows) != EXPECTED_COUNTS[split]:
            errors.append(f"expected {EXPECTED_COUNTS[split]} records, got {len(rows)}")
        if [row.get("image_id") for row in rows] != expected_ids:
            errors.append("image IDs or their order differ from the frozen manifest")
        total_objects = 0
        for row in rows:
            if set(row) != expected_fields:
                errors.append(f"image {row.get('image_id')}: unexpected record fields")
                continue
            if row["split"] != split or not isinstance(row["image_id"], int):
                errors.append(f"image {row['image_id']}: incorrect ID or split")
            if not isinstance(row["filename"], str) or not row["filename"]:
                errors.append(f"image {row['image_id']}: invalid filename")
            if not all(isinstance(row[k], int) and row[k] > 0 for k in ("width", "height")):
                errors.append(f"image {row['image_id']}: invalid dimensions")
            categories = set(row["categories"])
            if row["object_count"] != len(row["objects"]):
                errors.append(f"image {row['image_id']}: object_count mismatch")
            names = set()
            for obj in row["objects"]:
                if set(obj) != {"name", "bbox"} or not isinstance(obj.get("name"), str) or not obj["name"].strip():
                    errors.append(f"image {row['image_id']}: invalid object category")
                    continue
                names.add(obj["name"])
                box = obj["bbox"]
                if (not isinstance(box, list) or len(box) != 4
                        or any(not isinstance(value, (int, float)) for value in box)
                        or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                        or box[0] + box[2] > row["width"] + .02
                        or box[1] + box[3] > row["height"] + .02):
                    errors.append(f"image {row['image_id']}: invalid COCO bbox")
            if names != categories:
                errors.append(f"image {row['image_id']}: categories do not match object names")
            total_objects += len(row["objects"])
        summary[split] = {"images": len(rows), "objects": total_objects, "status": "PASS" if not errors else "FAIL", "errors": errors[:10]}
    return summary


def run(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    hashes_before = _hashes(root)
    manifests = {
        split: list(map(int, _read(root / "data" / "selected" / f"{split}_ids.json")["image_ids"]))
        for split in SPLITS
    }
    examples = {}
    for split in SPLITS:
        path = root / "data" / "metadata" / f"{split}.json"
        original = _read(path)
        if [int(row["image_id"]) for row in original] != manifests[split]:
            raise ValueError(f"Refusing to rewrite {path}: IDs/order differ from frozen manifest")
        if not original or "image_identification" not in original[0]:
            raise ValueError(f"{path} is already simplified or has an unexpected structure")
        examples.update({int(row["image_id"]): row for row in original if int(row["image_id"]) == 496571})
        _write(path, _simplify(split, original))

    summary = _validate(root, manifests)
    hashes_after = _hashes(root)
    unchanged = hashes_before == hashes_after
    if not unchanged:
        raise RuntimeError("A protected manifest, caption, embedding, or FAISS artifact changed")
    failures = [split for split, result in summary.items() if result["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"Simplified metadata validation failed: {failures}")
    return {
        "status": "PASS",
        "split_summary": summary,
        "protected_artifacts_unchanged": unchanged,
        "protected_artifact_count": len(hashes_after),
        "example_before": examples.get(496571),
        "example_after": next((row for split in SPLITS for row in _read(root / "data" / "metadata" / f"{split}.json") if row["image_id"] == 496571), None),
    }


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
