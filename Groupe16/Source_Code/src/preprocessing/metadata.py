import json
import re
from collections import Counter

from PIL import Image, ImageOps
from tqdm import tqdm

from src.data.splits import SPLIT_SOURCES, complexity, load_splits
from src.utils.core import digest, file_hash, path, read_json, stamp, write_json
from src.utils.geometry import bbox_status

LEXICONS = {
    "colour": ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray", "grey"],
    "shape": ["round", "circular", "square", "rectangular", "triangular"],
    "context": ["indoor", "indoors", "inside", "outdoor", "outdoors", "outside", "day", "daytime", "night", "nighttime"],
    "relation": ["on", "under", "above", "below", "behind", "beside", "near"],
}


def clean_caption(raw, lowercase=False):
    cleaned = " ".join(raw.split())
    return cleaned.lower() if lowercase else cleaned


def load_rgb(filename):
    with Image.open(filename) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Invalid image dimensions")
        return ImageOps.exif_transpose(image).convert("RGB")


def caption_attributes(captions):
    words = set(re.findall(r"[a-z]+", " ".join(captions).lower()))
    return {"source": "caption_heuristic", "method": "exact lowercase English word lexicons v1",
            "values": {key: sorted(words & set(terms)) for key, terms in LEXICONS.items()},
            "unknown_when_absent": True}


def spatial_relations(objects, threshold):
    """Aggregate ordered-box-pair relations in original annotation coordinates."""
    result = Counter()
    excluded = [obj["id"] for obj in objects if bbox_status(obj.get("bbox")) != "valid"]
    objects = [obj for obj in objects if bbox_status(obj.get("bbox")) == "valid"]
    for a in objects:
        ax, ay, aw, ah = a["bbox"]
        for b in objects:
            if a["id"] == b["id"]:
                continue
            bx, by, bw, bh = b["bbox"]
            if ax + aw / 2 < bx + bw / 2:
                result["left_of"] += 1
            if ay + ah / 2 < by + bh / 2:
                result["above"] += 1
            intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
            union = aw * ah + bw * bh - intersection
            if union > 0 and intersection / union >= threshold:
                result["overlap"] += 1
    result["right_of"] = result["left_of"]
    result["below"] = result["above"]
    return {"source": "bbox_heuristic", "counts": dict(result), "excluded_annotation_ids": excluded}


def prepare_metadata(cfg, data):
    from src.data.audit import audit
    audit(cfg, data)  # Never preprocess a dataset with critical unresolved anomalies.
    splits = load_splits(cfg)
    outputs = {}
    for split, ids in splits.items():
        source = SPLIT_SOURCES[split]
        records = []
        for image_id in tqdm(ids, desc=f"Metadata {split}", mininterval=5):
            image = data.captions[source].imgs[image_id]
            filename = data.image_path(source, image)
            rgb = load_rgb(filename)
            refs = data.references(source, image_id)
            objects = sorted(data.objects(source, image_id), key=lambda a: a["id"])
            records.append({
                "image_id": image_id, "file_name": image["file_name"], "split": split, "coco_source": source,
                "width": rgb.width, "height": rgb.height,
                "annotation_width": image["width"], "annotation_height": image["height"],
                "bbox_coordinates": "original COCO coordinates before EXIF transpose; overlay on original image",
                "file_sha256": file_hash(filename),
                "captions": [{"caption_id": c["id"], "caption_raw": c["caption"],
                              "caption_clean": clean_caption(c["caption"], cfg["preprocessing"]["lowercase"]),
                              "source": "coco_human_reference", "confidence": 1.0,
                              "confidence_policy": "human_reference_convention_only"} for c in refs],
                "objects": [{"id": a["id"], "category_id": a["category_id"], "category": data.categories[a["category_id"]]["name"],
                             "bbox": a["bbox"], "bbox_valid": bbox_status(a["bbox"]) == "valid",
                             "iscrowd": a.get("iscrowd", 0), "source": "coco_annotation"} for a in objects],
                "object_count": len(objects), "distinct_categories": len({a["category_id"] for a in objects}),
                "complexity": {"value": complexity(len(objects), cfg), "source": "instance_count_heuristic"},
                "derived_attributes": caption_attributes([c["caption"] for c in refs]),
                "reference_confidence_policy": {"human_caption_confidence": 1.0, "human_caption_threshold": 1.0,
                                                "meaning": "Retain all human references; not a calibrated probability or detector score. Derived heuristics have no confidence=1."},
                "spatial_relations": spatial_relations(objects, cfg["analysis"]["spatial_iou_threshold"]),
            })
            rgb.close()
        filename = path(cfg, "metadata") / f"{split}.jsonl"
        filename.parent.mkdir(parents=True, exist_ok=True)
        temporary = filename.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(filename)
        outputs[split] = {"rows": len(records), "sha256": file_hash(filename)}
    manifest = {**stamp(cfg), "outputs": outputs, "annotation_sha256": data.annotation_hashes,
                "split_sha256": digest(splits), "preprocessing": cfg["preprocessing"],
                "analysis": cfg["analysis"],
                "metadata_schema_version": 2,
                "metadata_policy": cfg.get("metadata", {}),
                "reference_policy": "Raw captions preserved. Only knowledge_base.jsonl may feed retrieval. Validation/test metadata is analysis-only.",
                "crowd_policy": "Counts are annotation instances including iscrowd; crowd annotations may represent multiple physical objects."}
    manifest["degenerate_bbox_policy"] = "Retain original object/category annotation and image; mark bbox_valid=false; exclude only from spatial relations and bounding-box drawing. See initial audit and documented annotation 918."
    write_json(path(cfg, "metadata") / "manifest.json", manifest)
    return manifest


def load_metadata(cfg, split):
    manifest = read_json(path(cfg, "metadata") / "manifest.json")
    filename = path(cfg, "metadata") / f"{split}.jsonl"
    if file_hash(filename) != manifest["outputs"][split]["sha256"]:
        raise ValueError(f"{split}: metadata changed; regenerate downstream artifacts")
    if manifest["split_sha256"] != digest(load_splits(cfg)):
        raise ValueError("Stale metadata: split IDs differ")
    if manifest["preprocessing"] != cfg["preprocessing"] or manifest["analysis"] != cfg["analysis"]:
        raise ValueError("Stale metadata: preprocessing/analysis configuration changed")
    if manifest.get("metadata_schema_version", 1) >= 2 and manifest.get("metadata_policy") != cfg.get("metadata", {}):
        raise ValueError("Stale metadata: human-reference policy changed")
    rows = [json.loads(line) for line in filename.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [r["image_id"] for r in rows] != load_splits(cfg)[split]:
        raise ValueError("Metadata ID order differs from frozen split")
    return rows
