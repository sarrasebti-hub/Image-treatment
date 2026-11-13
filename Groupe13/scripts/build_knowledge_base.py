
import json
import time
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_DIR = PROJECT_DIR / "data" / "embeddings"
INDEX_DIR = PROJECT_DIR / "data" / "index"

IMAGE_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "image_embeddings.npy"
)

IMAGE_IDS_FILE = (
    EMBEDDINGS_DIR / "image_ids.npy"
)

TEXT_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "text_embeddings.npy"
)

TEXT_IMAGE_IDS_FILE = (
    EMBEDDINGS_DIR / "text_image_ids.npy"
)

TEXT_ANNOTATION_IDS_FILE = (
    EMBEDDINGS_DIR / "text_annotation_ids.npy"
)

DIMENSION = 512


# ============================================================
# 2. LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print("\nLoading embeddings...")

    required_files = [
        IMAGE_EMBEDDINGS_FILE,
        IMAGE_IDS_FILE,
        TEXT_EMBEDDINGS_FILE,
        TEXT_IMAGE_IDS_FILE,
        TEXT_ANNOTATION_IDS_FILE
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    image_embeddings = np.load(
        IMAGE_EMBEDDINGS_FILE
    )

    image_ids = np.load(
        IMAGE_IDS_FILE
    )

    text_embeddings = np.load(
        TEXT_EMBEDDINGS_FILE
    )

    text_image_ids = np.load(
        TEXT_IMAGE_IDS_FILE
    )

    text_annotation_ids = np.load(
        TEXT_ANNOTATION_IDS_FILE
    )

    print("Image embeddings:", image_embeddings.shape)
    print("Image IDs:", image_ids.shape)

    print("Text embeddings:", text_embeddings.shape)
    print("Text image IDs:", text_image_ids.shape)
    print("Text annotation IDs:", text_annotation_ids.shape)

    return (
        image_embeddings,
        image_ids,
        text_embeddings,
        text_image_ids,
        text_annotation_ids
    )


# ============================================================
# 3. VALIDATE DATA
# ============================================================

def validate_data(
    image_embeddings,
    image_ids,
    text_embeddings,
    text_image_ids,
    text_annotation_ids
):

    print("\nValidating data...")

    if image_embeddings.ndim != 2:
        raise ValueError("Image embeddings must be 2D.")

    if text_embeddings.ndim != 2:
        raise ValueError("Text embeddings must be 2D.")

    if image_embeddings.shape[1] != DIMENSION:
        raise ValueError("Invalid image embedding dimension.")

    if text_embeddings.shape[1] != DIMENSION:
        raise ValueError("Invalid text embedding dimension.")

    if len(image_embeddings) != len(image_ids):
        raise ValueError("Image embeddings and IDs do not match.")

    if len(text_embeddings) != len(text_image_ids):
        raise ValueError("Text embeddings and image IDs do not match.")

    if len(text_embeddings) != len(text_annotation_ids):
        raise ValueError(
            "Text embeddings and annotation IDs do not match."
        )

    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image IDs detected.")

    if not np.isfinite(image_embeddings).all():
        raise ValueError("Invalid values in image embeddings.")

    if not np.isfinite(text_embeddings).all():
        raise ValueError("Invalid values in text embeddings.")

    # Ensure all caption image IDs belong to the Knowledge Base

    kb_id_set = set(image_ids.tolist())

    unknown_ids = set(text_image_ids.tolist()) - kb_id_set

    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids)} captions reference unknown images."
        )

    print("All validation checks passed.")


# ============================================================
# 4. NORMALIZE VECTORS
# ============================================================

def normalize_embeddings(
    image_embeddings,
    text_embeddings
):

    print("\nNormalizing embeddings...")

    image_embeddings = np.ascontiguousarray(
        image_embeddings,
        dtype=np.float32
    )

    text_embeddings = np.ascontiguousarray(
        text_embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(image_embeddings)
    faiss.normalize_L2(text_embeddings)

    print("Image vectors normalized.")
    print("Text vectors normalized.")

    return image_embeddings, text_embeddings


# ============================================================
# 5. BUILD IMAGE INDEX
# ============================================================

def build_image_index(image_embeddings):

    print("\nBuilding image FAISS index...")

    index = faiss.IndexFlatIP(DIMENSION)

    index.add(image_embeddings)

    print("Image index created.")
    print("Number of indexed images:", index.ntotal)

    return index


# ============================================================
# 6. BUILD TEXT INDEX
# ============================================================

def build_text_index(text_embeddings):

    print("\nBuilding text FAISS index...")

    index = faiss.IndexFlatIP(DIMENSION)

    index.add(text_embeddings)

    print("Text index created.")
    print("Number of indexed captions:", index.ntotal)

    return index


# ============================================================
# 7. SAVE KNOWLEDGE BASE
# ============================================================

def save_knowledge_base(
    image_index,
    text_index,
    image_ids,
    text_image_ids,
    text_annotation_ids
):

    print("\nSaving Knowledge Base...")

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS indexes

    faiss.write_index(
        image_index,
        str(INDEX_DIR / "image_index.faiss")
    )

    faiss.write_index(
        text_index,
        str(INDEX_DIR / "text_index.faiss")
    )

    # Save ID mappings

    np.save(
        INDEX_DIR / "image_index_ids.npy",
        image_ids
    )

    np.save(
        INDEX_DIR / "text_index_image_ids.npy",
        text_image_ids
    )

    np.save(
        INDEX_DIR / "text_index_annotation_ids.npy",
        text_annotation_ids
    )

    # Save information about the Knowledge Base

    metadata = {
        "model": "openai/clip-vit-base-patch32",
        "index_type": "IndexFlatIP",
        "similarity_metric": "Inner Product",
        "equivalent_metric": "Cosine Similarity for normalized vectors",
        "embedding_dimension": DIMENSION,
        "number_of_images": int(image_index.ntotal),
        "number_of_captions": int(text_index.ntotal),
        "image_index_file": "image_index.faiss",
        "text_index_file": "text_index.faiss",
        "image_id_mapping": "image_index_ids.npy",
        "text_image_id_mapping": "text_index_image_ids.npy",
        "text_annotation_id_mapping": "text_index_annotation_ids.npy"
    }

    with open(
        INDEX_DIR / "knowledge_base_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("All Knowledge Base files saved.")


# ============================================================
# 8. VERIFY INDEXES
# ============================================================

def verify_indexes(
    image_index,
    text_index
):

    print("\nVerifying FAISS indexes...")

    loaded_image_index = faiss.read_index(
        str(INDEX_DIR / "image_index.faiss")
    )

    loaded_text_index = faiss.read_index(
        str(INDEX_DIR / "text_index.faiss")
    )

    print(
        "Loaded image index:",
        loaded_image_index.ntotal
    )

    print(
        "Loaded text index:",
        loaded_text_index.ntotal
    )

    if loaded_image_index.ntotal != image_index.ntotal:
        raise RuntimeError("Image index verification failed.")

    if loaded_text_index.ntotal != text_index.ntotal:
        raise RuntimeError("Text index verification failed.")

    print("FAISS indexes verified successfully.")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("KNOWLEDGE BASE CONSTRUCTION")
    print("=" * 60)

    print("FAISS version:", faiss.__version__)
    print("Embedding dimension:", DIMENSION)

    # Load

    (
        image_embeddings,
        image_ids,
        text_embeddings,
        text_image_ids,
        text_annotation_ids
    ) = load_embeddings()

    # Validate

    validate_data(
        image_embeddings,
        image_ids,
        text_embeddings,
        text_image_ids,
        text_annotation_ids
    )

    # Normalize

    image_embeddings, text_embeddings = normalize_embeddings(
        image_embeddings,
        text_embeddings
    )

    # Build indexes

    image_index = build_image_index(
        image_embeddings
    )

    text_index = build_text_index(
        text_embeddings
    )

    # Save

    save_knowledge_base(
        image_index,
        text_index,
        image_ids,
        text_image_ids,
        text_annotation_ids
    )

    # Verify

    verify_indexes(
        image_index,
        text_index
    )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE CONSTRUCTION COMPLETED")
    print("=" * 60)

    print("Images indexed:", image_index.ntotal)
    print("Captions indexed:", text_index.ntotal)

    print(
        f"Execution time: {elapsed_time:.2f} seconds"
    )

    print("Output directory:", INDEX_DIR)


if __name__ == "__main__":
    main()
