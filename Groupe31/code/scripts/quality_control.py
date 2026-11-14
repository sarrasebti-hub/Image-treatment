

import json
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "kb": ROOT / "results/knowledge_base/knowledge_base.jsonl",
    "cleaned": ROOT / "results/knowledge_base/knowledge_base_cleaned_captions.jsonl",
    "metadata": ROOT / "results/metadata/image_metadata.jsonl",
    "images": ROOT / "results/preprocessed_images",
    "kb_ids": ROOT / "splits/knowledge_base_ids.json",
    "val_ids": ROOT / "splits/validation_ids.json",
    "test_ids": ROOT / "splits/test_ids.json",
}

report = {
    "checks": {},
    "errors": [],
    "warnings": [],
}

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def read_ids(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        for key in ("image_ids", "ids", "knowledge_base_ids",
                    "validation_ids", "test_ids"):
            if key in data and isinstance(data[key], list):
                return [str(x) for x in data[key]]

    raise ValueError(f"Unsupported ID file format: {path.name}")

def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        report["errors"].append(message)

print("=" * 60)
print("PHASE 1 - QUALITY CONTROL")
print("=" * 60)

# 1. Check required files
for name, path in PATHS.items():
    if name == "images":
        check(path.is_dir(), "Preprocessed images directory exists")
    else:
        check(path.is_file(), f"{name} file exists")

# Stop if required files are missing
missing = [
    name for name, path in PATHS.items()
    if not (path.is_dir() if name == "images" else path.is_file())
]

if missing:
    print("\nCannot continue. Some required files are missing.")
    raise SystemExit(1)

# 2. Load data
kb = read_jsonl(PATHS["kb"])
cleaned = read_jsonl(PATHS["cleaned"])
metadata = read_jsonl(PATHS["metadata"])

kb_ids = read_ids(PATHS["kb_ids"])
val_ids = read_ids(PATHS["val_ids"])
test_ids = read_ids(PATHS["test_ids"])

# 3. Duplicate ID checks
def duplicate_count(records):
    ids = [str(r["image_id"]) for r in records]
    return len(ids) - len(set(ids))

check(duplicate_count(kb) == 0, "No duplicate IDs in Knowledge Base")
check(duplicate_count(metadata) == 0, "No duplicate IDs in metadata")

check(len(kb_ids) == len(set(kb_ids)), "No duplicate IDs in KB split")
check(len(val_ids) == len(set(val_ids)), "No duplicate IDs in validation split")
check(len(test_ids) == len(set(test_ids)), "No duplicate IDs in test split")

# 4. Knowledge Base checks
check(len(kb) == 10000, "Knowledge Base contains 10,000 records")

captions_count = sum(len(r.get("captions", [])) for r in kb)
check(captions_count == 50000, "Knowledge Base contains 50,000 captions")

wrong_caption_count = sum(
    len(r.get("captions", [])) != 5 for r in kb
)
check(wrong_caption_count == 0, "Every KB image has exactly 5 captions")

kb_record_ids = {str(r["image_id"]) for r in kb}
check(kb_record_ids == set(kb_ids), "KB records match KB split IDs")

# 5. Caption consistency
cleaned_by_id = {
    str(r["image_id"]): r for r in cleaned
}

check(len(cleaned) == 10000, "Cleaned-caption file has 10,000 records")
check(
    set(cleaned_by_id) == kb_record_ids,
    "Cleaned-caption IDs match Knowledge Base IDs"
)

caption_mismatches = 0
invalid_cleaned_records = 0

for record in kb:
    image_id = str(record["image_id"])
    cleaned_record = cleaned_by_id.get(image_id)

    if cleaned_record is None:
        caption_mismatches += 1
        continue

    original = record.get("captions", [])
    cleaned_original = cleaned_record.get("original_captions", [])

    if original != cleaned_original:
        caption_mismatches += 1

    if len(cleaned_record.get("cleaned_captions", [])) != 5:
        invalid_cleaned_records += 1

check(caption_mismatches == 0, "Original captions match cleaned-file references")
check(invalid_cleaned_records == 0, "Every image has 5 cleaned captions")

# 6. Metadata checks
metadata_by_id = {
    str(r["image_id"]): r for r in metadata
}

check(len(metadata) == 10000, "Metadata contains 10,000 records")
check(
    set(metadata_by_id) == kb_record_ids,
    "Metadata IDs match Knowledge Base IDs"
)

required_fields = [
    "image_id",
    "file_name",
    "objects",
    "object_categories",
    "spatial_relationships",
    "caption_mentions",
]

invalid_metadata = 0

for record in metadata:
    if any(field not in record for field in required_fields):
        invalid_metadata += 1

check(invalid_metadata == 0, "All metadata records contain required fields")

# 7. Image integrity
image_files = list(PATHS["images"].glob("*.jpg"))
image_ids_without_files = []
unreadable_images = []

kb_by_id = {str(r["image_id"]): r for r in kb}

for image_id, record in kb_by_id.items():
    image_path = PATHS["images"] / record["file_name"]

    if not image_path.is_file():
        image_ids_without_files.append(image_id)
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        unreadable_images.append(image_id)

check(len(image_files) == 10000, "Preprocessed folder contains 10,000 JPG files")
check(len(image_ids_without_files) == 0, "All KB images have preprocessed files")
check(len(unreadable_images) == 0, "All preprocessed images are readable")

# 8. Split integrity and anti-leakage
kb_set = set(kb_ids)
val_set = set(val_ids)
test_set = set(test_ids)

kb_val_overlap = kb_set & val_set
kb_test_overlap = kb_set & test_set
val_test_overlap = val_set & test_set

check(len(kb_val_overlap) == 0, "No overlap between KB and validation")
check(len(kb_test_overlap) == 0, "No overlap between KB and test")
check(len(val_test_overlap) == 0, "No overlap between validation and test")

check(len(kb_ids) == 10000, "KB split contains 10,000 IDs")
check(len(val_ids) == 100, "Validation split contains 100 IDs")
check(len(test_ids) == 1000, "Test split contains 1,000 IDs")

# 9. Save report
report["checks"] = {
    "knowledge_base_records": len(kb),
    "knowledge_base_captions": captions_count,
    "metadata_records": len(metadata),
    "cleaned_caption_records": len(cleaned),
    "preprocessed_images": len(image_files),
    "kb_split_ids": len(kb_ids),
    "validation_ids": len(val_ids),
    "test_ids": len(test_ids),
    "caption_mismatches": caption_mismatches,
    "invalid_cleaned_records": invalid_cleaned_records,
    "invalid_metadata_records": invalid_metadata,
    "missing_images": len(image_ids_without_files),
    "unreadable_images": len(unreadable_images),
    "kb_validation_overlap": len(kb_val_overlap),
    "kb_test_overlap": len(kb_test_overlap),
    "validation_test_overlap": len(val_test_overlap),
}

output_dir = ROOT / "results/quality_control"
output_dir.mkdir(parents=True, exist_ok=True)

report["status"] = "PASSED" if not report["errors"] else "FAILED"

report_path = output_dir / "quality_control_report.json"
report_path.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("QUALITY CONTROL SUMMARY")
print("=" * 60)
print(f"Errors: {len(report['errors'])}")
print(f"Warnings: {len(report['warnings'])}")
print(f"Status: {report['status']}")
print(f"Report saved: {report_path}")
print("=" * 60)
