
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

SPLITS_DIR = DATA_DIR / "splits"
CAPTIONS_DIR = DATA_DIR / "captions"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEX_DIR = DATA_DIR / "index"
IMAGES_DIR = DATA_DIR / "processed_images"
REPORTS_DIR = PROJECT_DIR / "reports"

# Expected dataset sizes

EXPECTED_KB_IMAGES = 10000
EXPECTED_VALIDATION_IMAGES = 100
EXPECTED_TEST_IMAGES = 100

EXPECTED_EMBEDDING_DIMENSION = 512

# Input files

KB_IDS_FILE = SPLITS_DIR / "knowledge_base_image_ids.json"
VALIDATION_IDS_FILE = SPLITS_DIR / "validation_image_ids.json"
TEST_IDS_FILE = SPLITS_DIR / "test_image_ids.json"

CAPTIONS_FILE = CAPTIONS_DIR / "kb_captions_cleaned.json"

IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "image_embeddings.npy"
IMAGE_IDS_FILE = EMBEDDINGS_DIR / "image_ids.npy"

TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "text_embeddings.npy"
TEXT_IMAGE_IDS_FILE = EMBEDDINGS_DIR / "text_image_ids.npy"
TEXT_ANNOTATION_IDS_FILE = (
    EMBEDDINGS_DIR / "text_annotation_ids.npy"
)

# FAISS files

IMAGE_INDEX_FILE = INDEX_DIR / "image_index.faiss"
TEXT_INDEX_FILE = INDEX_DIR / "text_index.faiss"

# Report

REPORT_FILE = REPORTS_DIR / "quality_control_report.json"


# ============================================================
# 2. REPORT STRUCTURE
# ============================================================

report = {
    "project": "COCO Image Captioning - RAG",
    "checks": {},
    "statistics": {},
    "warnings": [],
    "errors": [],
    "summary": {}
}


def add_check(name, status, details=None):

    report["checks"][name] = {
        "status": status,
        "details": details or {}
    }

    if status == "FAIL":
        report["errors"].append(name)

    elif status == "WARNING":
        report["warnings"].append(name)


# ============================================================
# 3. JSON LOADING
# ============================================================

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ids(data):

    if isinstance(data, list):
        values = data

    elif isinstance(data, dict):

        values = data.get(
            "image_ids",
            data.get(
                "knowledge_base_image_ids",
                data.get("ids")
            )
        )

        if values is None:
            raise ValueError(
                "Cannot identify image IDs in JSON file."
            )

    else:
        raise ValueError("Unsupported JSON structure.")

    result = []

    for item in values:

        if isinstance(item, dict):
            item = item.get(
                "image_id",
                item.get("id")
            )

        if item is not None:
            result.append(int(item))

    return result


# ============================================================
# 4. CAPTION EXTRACTION
# ============================================================

def extract_captions(data):

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):

        if isinstance(data.get("captions"), list):
            records = data["captions"]

        elif isinstance(data.get("annotations"), list):
            records = data["annotations"]

        elif isinstance(data.get("images"), list):

            records = []

            for image in data["images"]:

                image_id = image.get(
                    "image_id",
                    image.get("id")
                )

                captions = image.get(
                    "captions",
                    image.get("cleaned_captions", [])
                )

                for caption in captions:

                    if isinstance(caption, dict):
                        record = caption.copy()
                        record.setdefault("image_id", image_id)

                    else:
                        record = {
                            "image_id": image_id,
                            "caption": caption
                        }

                    records.append(record)

        else:
            raise ValueError(
                "Cannot identify caption records."
            )

    else:
        raise ValueError("Unsupported caption JSON structure.")

    extracted = []

    for position, item in enumerate(records):

        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid caption record at position {position}."
            )

        image_id = item.get(
            "image_id",
            item.get("id")
        )

        caption = item.get(
            "cleaned_caption",
            item.get("caption", item.get("text"))
        )

        annotation_id = item.get(
            "annotation_id",
            item.get("id")
        )

        if image_id is None or caption is None:
            raise ValueError(
                f"Missing image ID or caption at position {position}."
            )

        extracted.append({
            "image_id": int(image_id),
            "caption": str(caption).strip(),
            "annotation_id": (
                int(annotation_id)
                if annotation_id is not None
                else None
            )
        })

    return extracted


# ============================================================
# 5. CHECK SPLITS
# ============================================================

def check_splits():

    print("\n" + "=" * 60)
    print("1. DATA SPLIT CHECK")
    print("=" * 60)

    try:

        kb_ids = extract_ids(
            load_json(KB_IDS_FILE)
        )

        validation_ids = extract_ids(
            load_json(VALIDATION_IDS_FILE)
        )

        test_ids = extract_ids(
            load_json(TEST_IDS_FILE)
        )

        kb_set = set(kb_ids)
        validation_set = set(validation_ids)
        test_set = set(test_ids)

        duplicate_kb = len(kb_ids) - len(kb_set)
        duplicate_validation = (
            len(validation_ids) - len(validation_set)
        )
        duplicate_test = len(test_ids) - len(test_set)

        kb_validation_overlap = kb_set & validation_set
        kb_test_overlap = kb_set & test_set
        validation_test_overlap = validation_set & test_set

        details = {
            "knowledge_base_count": len(kb_ids),
            "validation_count": len(validation_ids),
            "test_count": len(test_ids),
            "duplicate_kb_ids": duplicate_kb,
            "duplicate_validation_ids": duplicate_validation,
            "duplicate_test_ids": duplicate_test,
            "kb_validation_overlap": len(kb_validation_overlap),
            "kb_test_overlap": len(kb_test_overlap),
            "validation_test_overlap": len(validation_test_overlap)
        }

        valid = (
            len(kb_ids) == EXPECTED_KB_IMAGES
            and len(validation_ids) == EXPECTED_VALIDATION_IMAGES
            and len(test_ids) == EXPECTED_TEST_IMAGES
            and duplicate_kb == 0
            and duplicate_validation == 0
            and duplicate_test == 0
            and len(kb_validation_overlap) == 0
            and len(kb_test_overlap) == 0
            and len(validation_test_overlap) == 0
        )

        status = "PASS" if valid else "FAIL"

        add_check(
            "data_splits",
            status,
            details
        )

        report["statistics"]["knowledge_base_images"] = len(kb_ids)
        report["statistics"]["validation_images"] = len(validation_ids)
        report["statistics"]["test_images"] = len(test_ids)

        print("Knowledge Base:", len(kb_ids))
        print("Validation:", len(validation_ids))
        print("Test:", len(test_ids))
        print("KB / Validation overlap:", len(kb_validation_overlap))
        print("KB / Test overlap:", len(kb_test_overlap))
        print("Validation / Test overlap:", len(validation_test_overlap))
        print("Status:", status)

        return kb_ids, validation_ids, test_ids

    except Exception as error:

        add_check(
            "data_splits",
            "FAIL",
            {"error": str(error)}
        )

        print("ERROR:", error)

        return [], [], []


# ============================================================
# 6. CHECK EMBEDDING FILES
# ============================================================

def check_embedding_files():

    print("\n" + "=" * 60)
    print("2. EMBEDDING FILE CHECK")
    print("=" * 60)

    files = {
        "image_embeddings": IMAGE_EMBEDDINGS_FILE,
        "image_ids": IMAGE_IDS_FILE,
        "text_embeddings": TEXT_EMBEDDINGS_FILE,
        "text_image_ids": TEXT_IMAGE_IDS_FILE,
        "text_annotation_ids": TEXT_ANNOTATION_IDS_FILE
    }

    missing = []

    for name, path in files.items():

        if not path.exists():
            missing.append(name)
            print("MISSING:", name)

        else:
            print("FOUND:", name)

    if missing:

        add_check(
            "embedding_files",
            "FAIL",
            {"missing_files": missing}
        )

        return None

    add_check(
        "embedding_files",
        "PASS",
        {"files_found": len(files)}
    )

    try:

        arrays = {
            name: np.load(path, mmap_mode="r")
            for name, path in files.items()
        }

        return arrays

    except Exception as error:

        add_check(
            "embedding_loading",
            "FAIL",
            {"error": str(error)}
        )

        return None


# ============================================================
# 7. CHECK EMBEDDING DIMENSIONS AND VALUES
# ============================================================

def check_embedding_quality(arrays):

    print("\n" + "=" * 60)
    print("3. EMBEDDING QUALITY CHECK")
    print("=" * 60)

    if arrays is None:
        return

    image_embeddings = arrays["image_embeddings"]
    image_ids = arrays["image_ids"]

    text_embeddings = arrays["text_embeddings"]
    text_image_ids = arrays["text_image_ids"]
    annotation_ids = arrays["text_annotation_ids"]

    details = {
        "image_embedding_shape": list(image_embeddings.shape),
        "text_embedding_shape": list(text_embeddings.shape),
        "image_embedding_dtype": str(image_embeddings.dtype),
        "text_embedding_dtype": str(text_embeddings.dtype)
    }

    shape_valid = (
        image_embeddings.ndim == 2
        and text_embeddings.ndim == 2
        and image_embeddings.shape[1] == EXPECTED_EMBEDDING_DIMENSION
        and text_embeddings.shape[1] == EXPECTED_EMBEDDING_DIMENSION
        and len(image_embeddings) == len(image_ids)
        and len(text_embeddings) == len(text_image_ids)
        and len(text_embeddings) == len(annotation_ids)
    )

    add_check(
        "embedding_dimensions",
        "PASS" if shape_valid else "FAIL",
        details
    )

    print("Image shape:", image_embeddings.shape)
    print("Text shape:", text_embeddings.shape)
    print("Dimensions status:", "PASS" if shape_valid else "FAIL")

    # Check NaN and infinite values

    image_invalid = 0
    text_invalid = 0

    for start in range(0, len(image_embeddings), 1000):

        batch = np.asarray(
            image_embeddings[start:start + 1000]
        )

        image_invalid += int(
            np.size(batch) - np.count_nonzero(np.isfinite(batch))
        )

    for start in range(0, len(text_embeddings), 1000):

        batch = np.asarray(
            text_embeddings[start:start + 1000]
        )

        text_invalid += int(
            np.size(batch) - np.count_nonzero(np.isfinite(batch))
        )

    finite_valid = (
        image_invalid == 0
        and text_invalid == 0
    )

    add_check(
        "nan_and_infinite_values",
        "PASS" if finite_valid else "FAIL",
        {
            "invalid_image_values": image_invalid,
            "invalid_text_values": text_invalid
        }
    )

    print("Invalid image values:", image_invalid)
    print("Invalid text values:", text_invalid)

    # Check duplicate IDs

    duplicate_image_ids = (
        len(image_ids) - len(np.unique(image_ids))
    )

    duplicate_annotation_ids = (
        len(annotation_ids) - len(np.unique(annotation_ids))
    )

    add_check(
        "duplicate_ids",
        "PASS" if (
            duplicate_image_ids == 0
            and duplicate_annotation_ids == 0
        ) else "FAIL",
        {
            "duplicate_image_ids": int(duplicate_image_ids),
            "duplicate_annotation_ids": int(
                duplicate_annotation_ids
            )
        }
    )

    print("Duplicate image IDs:", duplicate_image_ids)
    print("Duplicate annotation IDs:", duplicate_annotation_ids)

    # Check vector normalization

    image_norms = []
    text_norms = []

    for start in range(0, len(image_embeddings), 1000):

        batch = np.asarray(
            image_embeddings[start:start + 1000],
            dtype=np.float32
        )

        image_norms.extend(
            np.linalg.norm(batch, axis=1).tolist()
        )

    for start in range(0, len(text_embeddings), 1000):

        batch = np.asarray(
            text_embeddings[start:start + 1000],
            dtype=np.float32
        )

        text_norms.extend(
            np.linalg.norm(batch, axis=1).tolist()
        )

    image_norms = np.asarray(image_norms)
    text_norms = np.asarray(text_norms)

    normalized = (
        np.allclose(image_norms, 1.0, atol=1e-3)
        and np.allclose(text_norms, 1.0, atol=1e-3)
    )

    add_check(
        "l2_normalization",
        "PASS" if normalized else "WARNING",
        {
            "image_norm_min": float(image_norms.min()),
            "image_norm_max": float(image_norms.max()),
            "text_norm_min": float(text_norms.min()),
            "text_norm_max": float(text_norms.max())
        }
    )

    print("L2 normalization:", "PASS" if normalized else "WARNING")

    report["statistics"]["image_embeddings"] = {
        "count": int(len(image_embeddings)),
        "dimension": int(image_embeddings.shape[1])
    }

    report["statistics"]["text_embeddings"] = {
        "count": int(len(text_embeddings)),
        "dimension": int(text_embeddings.shape[1])
    }


# ============================================================
# 8. CHECK CAPTION / IMAGE MATCHING
# ============================================================

def check_caption_matching(
    kb_ids,
    arrays
):

    print("\n" + "=" * 60)
    print("4. CAPTION AND IMAGE MATCHING")
    print("=" * 60)

    if arrays is None:
        return

    try:

        captions = extract_captions(
            load_json(CAPTIONS_FILE)
        )

        kb_set = set(kb_ids)

        caption_image_ids = [
            item["image_id"] for item in captions
        ]

        caption_counts = Counter(caption_image_ids)

        missing_caption_images = (
            kb_set - set(caption_image_ids)
        )

        unknown_caption_images = (
            set(caption_image_ids) - kb_set
        )

        embedding_image_ids = set(
            arrays["image_ids"].tolist()
        )

        text_image_ids = set(
            arrays["text_image_ids"].tolist()
        )

        missing_image_embeddings = (
            kb_set - embedding_image_ids
        )

        missing_text_embeddings = (
            kb_set - text_image_ids
        )

        captions_without_text_embeddings = (
            set(caption_image_ids) - text_image_ids
        )

        details = {
            "caption_records": len(captions),
            "images_with_captions": len(caption_counts),
            "missing_caption_images": len(missing_caption_images),
            "unknown_caption_images": len(unknown_caption_images),
            "missing_image_embeddings": len(missing_image_embeddings),
            "missing_text_embeddings": len(missing_text_embeddings),
            "captions_without_text_embeddings": len(
                captions_without_text_embeddings
            ),
            "minimum_captions_per_image": (
                min(caption_counts.values())
                if caption_counts else 0
            ),
            "maximum_captions_per_image": (
                max(caption_counts.values())
                if caption_counts else 0
            )
        }

        valid = (
            len(missing_caption_images) == 0
            and len(unknown_caption_images) == 0
            and len(missing_image_embeddings) == 0
            and len(missing_text_embeddings) == 0
            and len(captions_without_text_embeddings) == 0
        )

        add_check(
            "caption_image_matching",
            "PASS" if valid else "FAIL",
            details
        )

        report["statistics"]["caption_records"] = len(captions)

        print("Caption records:", len(captions))
        print("Images with captions:", len(caption_counts))
        print("Missing caption images:", len(missing_caption_images))
        print("Unknown caption images:", len(unknown_caption_images))
        print("Missing image embeddings:", len(missing_image_embeddings))
        print("Missing text embeddings:", len(missing_text_embeddings))

        print("Status:", "PASS" if valid else "FAIL")

        # Additional caption count information

        unusual_counts = {
            str(image_id): count
            for image_id, count in caption_counts.items()
            if count != 5
        }

        report["statistics"]["images_with_caption_count_other_than_5"] = (
            len(unusual_counts)
        )

        report["statistics"]["caption_count_distribution"] = {
            str(count): sum(
                1 for value in caption_counts.values()
                if value == count
            )
            for count in sorted(set(caption_counts.values()))
        }

        if unusual_counts:
            print(
                "Images with a caption count different from 5:",
                len(unusual_counts)
            )

    except Exception as error:

        add_check(
            "caption_image_matching",
            "FAIL",
            {"error": str(error)}
        )

        print("ERROR:", error)


# ============================================================
# 9. CHECK PROCESSED IMAGES
# ============================================================

def check_processed_images(kb_ids):

    print("\n" + "=" * 60)
    print("5. PROCESSED IMAGE CHECK")
    print("=" * 60)

    if not IMAGES_DIR.exists():

        add_check(
            "processed_images",
            "WARNING",
            {"message": "Processed images directory not found."}
        )

        print("Processed images directory not found.")
        return

    missing = []

    for image_id in kb_ids:

        filename = f"{image_id:012d}"

        possible_files = [
            IMAGES_DIR / f"{filename}.jpg",
            IMAGES_DIR / f"{filename}.jpeg",
            IMAGES_DIR / f"{filename}.png"
        ]

        if not any(path.exists() for path in possible_files):
            missing.append(image_id)

    valid = len(missing) == 0

    add_check(
        "processed_images",
        "PASS" if valid else "FAIL",
        {
            "expected_images": len(kb_ids),
            "missing_images": len(missing),
            "first_missing_ids": missing[:20]
        }
    )

    print("Expected images:", len(kb_ids))
    print("Missing images:", len(missing))
    print("Status:", "PASS" if valid else "FAIL")


# ============================================================
# 10. CHECK FAISS INDEXES
# ============================================================

def check_faiss_indexes(arrays):

    print("\n" + "=" * 60)
    print("6. FAISS INDEX CHECK")
    print("=" * 60)

    if not IMAGE_INDEX_FILE.exists() or not TEXT_INDEX_FILE.exists():

        add_check(
            "faiss_indexes",
            "WARNING",
            {
                "message": "FAISS index files are not available yet."
            }
        )

        print("FAISS indexes not found. Build them first.")
        return

    try:

        import faiss

        image_index = faiss.read_index(
            str(IMAGE_INDEX_FILE)
        )

        text_index = faiss.read_index(
            str(TEXT_INDEX_FILE)
        )

        expected_images = len(arrays["image_ids"])
        expected_texts = len(arrays["text_image_ids"])

        image_count_valid = (
            image_index.ntotal == expected_images
        )

        text_count_valid = (
            text_index.ntotal == expected_texts
        )

        valid = (
            image_count_valid
            and text_count_valid
            and image_index.d == EXPECTED_EMBEDDING_DIMENSION
            and text_index.d == EXPECTED_EMBEDDING_DIMENSION
        )

        details = {
            "faiss_version": faiss.__version__,
            "image_index_vectors": int(image_index.ntotal),
            "text_index_vectors": int(text_index.ntotal),
            "image_index_dimension": int(image_index.d),
            "text_index_dimension": int(text_index.d),
            "expected_image_vectors": int(expected_images),
            "expected_text_vectors": int(expected_texts)
        }

        add_check(
            "faiss_indexes",
            "PASS" if valid else "FAIL",
            details
        )

        print("FAISS version:", faiss.__version__)
        print("Image index vectors:", image_index.ntotal)
        print("Text index vectors:", text_index.ntotal)
        print("Status:", "PASS" if valid else "FAIL")

    except Exception as error:

        add_check(
            "faiss_indexes",
            "FAIL",
            {"error": str(error)}
        )

        print("FAISS ERROR:", error)


# ============================================================
# 11. CHECK DATA LEAKAGE
# ============================================================

def check_data_leakage(
    kb_ids,
    validation_ids,
    test_ids
):

    print("\n" + "=" * 60)
    print("7. DATA LEAKAGE CHECK")
    print("=" * 60)

    if not kb_ids or not validation_ids or not test_ids:

        add_check(
            "data_leakage",
            "FAIL",
            {"message": "Cannot check leakage because splits are unavailable."}
        )

        return

    kb_set = set(kb_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    kb_validation = kb_set & validation_set
    kb_test = kb_set & test_set
    validation_test = validation_set & test_set

    # Check the image IDs represented by the text embeddings

    text_leakage = set()

    if TEXT_IMAGE_IDS_FILE.exists():

        text_ids = set(
            np.load(TEXT_IMAGE_IDS_FILE).tolist()
        )

        text_leakage = text_ids & (
            validation_set | test_set
        )

    # Check that reference captions from validation/test
    # have not been included in the KB caption file.

    valid = (
        len(kb_validation) == 0
        and len(kb_test) == 0
        and len(validation_test) == 0
        and len(text_leakage) == 0
    )

    details = {
        "kb_validation_overlap": len(kb_validation),
        "kb_test_overlap": len(kb_test),
        "validation_test_overlap": len(validation_test),
        "validation_test_ids_in_text_embeddings": len(text_leakage)
    }

    add_check(
        "data_leakage",
        "PASS" if valid else "FAIL",
        details
    )

    print("KB / Validation overlap:", len(kb_validation))
    print("KB / Test overlap:", len(kb_test))
    print("Validation / Test overlap:", len(validation_test))
    print("Validation/Test IDs in text embeddings:", len(text_leakage))
    print("Status:", "PASS" if valid else "FAIL")


# ============================================================
# 12. SAVE QUALITY CONTROL REPORT
# ============================================================

def save_report():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_checks = len(report["checks"])

    passed = sum(
        1 for item in report["checks"].values()
        if item["status"] == "PASS"
    )

    failed = sum(
        1 for item in report["checks"].values()
        if item["status"] == "FAIL"
    )

    warnings = sum(
        1 for item in report["checks"].values()
        if item["status"] == "WARNING"
    )

    report["summary"] = {
        "total_checks": total_checks,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "overall_status": (
            "PASS" if failed == 0 else "FAIL"
        )
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("QUALITY CONTROL SUMMARY")
    print("=" * 60)

    print("Total checks:", total_checks)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Warnings:", warnings)

    print(
        "Overall status:",
        report["summary"]["overall_status"]
    )

    print("\nReport saved at:")
    print(REPORT_FILE)


# ============================================================
# 13. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("COCO DATASET - QUALITY CONTROL")
    print("=" * 60)

    # 1. Check splits

    kb_ids, validation_ids, test_ids = check_splits()

    # 2. Check embedding files

    arrays = check_embedding_files()

    # 3. Check embedding quality

    check_embedding_quality(arrays)

    # 4. Check caption/image matching

    if kb_ids:
        check_caption_matching(kb_ids, arrays)

        # 5. Check processed images

        check_processed_images(kb_ids)

    # 6. Check FAISS indexes

    if arrays is not None:
        check_faiss_indexes(arrays)

    # 7. Check data leakage

    check_data_leakage(
        kb_ids,
        validation_ids,
        test_ids
    )

    # 8. Save report

    save_report()

    elapsed = time.time() - start_time

    print("\nExecution time: {:.2f} seconds".format(elapsed))
    print("\nQUALITY CONTROL COMPLETED")


if __name__ == "__main__":
    main()
