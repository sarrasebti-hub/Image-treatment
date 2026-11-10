import csv
from collections import Counter, defaultdict

from PIL import Image
from tqdm import tqdm

from src.data.splits import SPLIT_SOURCES, load_splits
from src.utils.core import digest, file_hash, path, read_json, stamp, write_json
from src.utils.geometry import bbox_status


def audit(cfg, data):
    """Write the report before raising. No sample is deleted or silently excluded."""
    issues, rows = [], []
    def issue(kind, split=None, image_id=None, detail="", severity="error"):
        issues.append(dict(kind=kind, split=split, image_id=image_id, detail=detail, severity=severity))
    for error in data.validation_errors:
        issue("annotation_integrity", detail=error)
    try:
        splits = load_splits(cfg)
    except (ValueError, FileNotFoundError) as exc:
        issue("split_integrity", detail=str(exc))
        write_json(path(cfg, "reports") / "audit.json", {"passed": False, "issues": issues})
        raise ValueError("Split audit failed; see reports/audit.json") from exc
    manifest = read_json(path(cfg, "splits") / "split_manifest.json")
    if data.annotation_hashes != manifest["annotation_sha256"]:
        issue("annotations_changed", detail="Annotation files differ from frozen split manifest")
    file_hashes, pixel_hashes, caption_owners = defaultdict(list), defaultdict(list), defaultdict(set)
    actual = Counter()
    caption_ids = defaultdict(list)
    for split, ids in splits.items():
        source = SPLIT_SOURCES[split]
        for image_id in tqdm(ids, desc=f"Audit {split}", mininterval=5):
            record = dict(split=split, image_id=image_id, exists=False, readable=False, width=None, height=None, mode=None)
            image = data.captions[source].imgs.get(image_id)
            if image is None:
                issue("unknown_image_id", split, image_id)
                rows.append(record)
                continue
            refs = data.references(source, image_id)
            record["caption_count"] = len(refs)
            if len(refs) != cfg["expected_captions"]:
                issue("caption_count", split, image_id, str(len(refs)))
            for ref in refs:
                caption_ids[ref["id"]].append((split, image_id))
                if not isinstance(ref.get("caption"), str) or not ref["caption"].strip():
                    issue("missing_caption", split, image_id, str(ref["id"]))
                else:
                    caption_owners[" ".join(ref["caption"].lower().split())].add((split, image_id, ref["id"]))
            obj_image = data.instances[source].imgs.get(image_id)
            if obj_image is None:
                issue("missing_instance_image", split, image_id)
            elif any(obj_image[k] != image[k] for k in ("width", "height", "file_name")):
                issue("annotation_image_mismatch", split, image_id)
            for obj in data.objects(source, image_id):
                if obj["category_id"] not in data.categories:
                    issue("unknown_category", split, image_id, str(obj["id"]))
                box = obj.get("bbox", [])
                status = bbox_status(box)
                if status in ("malformed", "negative"):
                    issue("invalid_bbox", split, image_id, str(obj["id"]))
                elif status == "degenerate":
                    issue("degenerate_bbox", split, image_id,
                          f"annotation_id={obj['id']}; bbox={box}; retained raw/category annotation; excluded only from geometric relations/box drawing", severity="warning")
            filename = data.image_path(source, image)
            record["exists"] = filename.is_file()
            if not record["exists"]:
                issue("missing_image", split, image_id, image["file_name"])
            else:
                actual[split] += 1
                try:
                    with Image.open(filename) as img:
                        img.verify()
                    with Image.open(filename) as img:
                        img.load()
                        record.update(width=img.width, height=img.height, mode=img.mode, readable=True)
                        if img.size != (image["width"], image["height"]):
                            issue("dimensions_mismatch", split, image_id, str(img.size))
                        if cfg["audit"]["file_hashes"]:
                            file_hashes[file_hash(filename)].append((split, image_id))
                            # Include dimensions: equal byte payload alone is not a decoded-image identity.
                            pixels = img.convert("RGB")
                            import hashlib
                            pixel_hash = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
                            pixel_hashes[pixel_hash].append((split, image_id))
                except (OSError, ValueError) as exc:
                    issue("unreadable_image", split, image_id, str(exc))
            rows.append(record)
    for method, hashes in (("file", file_hashes), ("decoded_pixels", pixel_hashes)):
        for owners in hashes.values():
            if len(owners) > 1:
                cross_split = len({owner[0] for owner in owners}) > 1
                issue(f"duplicate_{method}", detail=str(owners), severity="error" if cross_split else "warning")
    for caption_id, owners in caption_ids.items():
        if len(owners) > 1:
            issue("duplicate_caption_id", detail=f"{caption_id}: {owners}")
    shared_details = [{"caption_normalized": caption, "occurrences": [
        {"split": owner[0], "image_id": owner[1], "caption_id": owner[2]} for owner in sorted(owners)]}
        for caption, owners in caption_owners.items() if len({owner[0] for owner in owners}) > 1]
    shared_text = len(shared_details)
    if shared_text:
        issue("shared_caption_text", detail=f"{shared_text} identical normalized caption strings across splits; inspect, not proof of provenance leakage", severity="warning")
    report = {**stamp(cfg), "passed": not any(i["severity"] == "error" for i in issues),
              "expected_images": {k: len(v) for k, v in splits.items()}, "present_images": dict(actual),
              "readable_images": dict(Counter(r["split"] for r in rows if r["readable"])),
              "issues": issues, "images": rows, "shared_caption_details": shared_details, "annotation_sha256": data.annotation_hashes,
              "split_sha256": digest(splits), "removals": [],
              "duplicate_scope": "Exact file and decoded RGB pixels; perceptual near-duplicates not checked"}
    write_json(path(cfg, "reports") / "audit.json", report)
    filename = path(cfg, "tables") / "audit_issues.csv"
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["kind", "split", "image_id", "detail", "severity"])
        writer.writeheader()
        writer.writerows(issues)
    with (filename.parent / "shared_caption_strings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["caption_normalized", "split", "image_id", "caption_id"])
        writer.writeheader()
        for item in shared_details:
            writer.writerows({"caption_normalized": item["caption_normalized"], **owner} for owner in item["occurrences"])
    if not report["passed"]:
        raise ValueError("Critical data audit errors; see reports/audit.json. No samples removed.")
    return report
