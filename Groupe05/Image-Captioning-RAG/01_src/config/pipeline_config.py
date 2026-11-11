import json
from pathlib import Path

# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

# This file is located in 01_src/config/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DATA = PROJECT_ROOT / "00_original_data"
ORIGINAL_ANNOTATIONS = ORIGINAL_DATA / "annotations"

PROCESSED_DATA = PROJECT_ROOT / "02_processed_data"
SPLITS_DIR = PROCESSED_DATA / "splits"


# ============================================================
# 2. DATASET SOURCES
# ============================================================

TRAIN_SOURCE = "train2017"
TEST_SOURCE = "val2017"

TRAIN_IMAGES_DIR = ORIGINAL_DATA / TRAIN_SOURCE
TEST_IMAGES_DIR = ORIGINAL_DATA / TEST_SOURCE

TRAIN_CAPTIONS_FILE = (
    ORIGINAL_ANNOTATIONS / "captions_train2017.json"
)
TEST_CAPTIONS_FILE = (
    ORIGINAL_ANNOTATIONS / "captions_val2017.json"
)

TRAIN_INSTANCES_FILE = (
    ORIGINAL_ANNOTATIONS / "instances_train2017.json"
)
TEST_INSTANCES_FILE = (
    ORIGINAL_ANNOTATIONS / "instances_val2017.json"
)


# ============================================================
# 3. SPLIT CONFIGURATION
# ============================================================

RANDOM_SEED = 45

TRAINING_SIZE = 9900
VALIDATION_SIZE = 100
TEST_SIZE = 100

# Automatically calculated: training + validation
TOTAL_TRAIN_POOL = TRAINING_SIZE + VALIDATION_SIZE

# Number of captions retained per image
CAPTIONS_PER_IMAGE = 5


# ============================================================
# 4. EMBEDDING CONFIGURATION
# ============================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

SBERT_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CLIP_BATCH_SIZE = 32
SBERT_BATCH_SIZE = 64

EMBEDDING_DTYPE = "float32"
NORMALIZE_EMBEDDINGS = True


# ============================================================
# 5. RETRIEVAL CONFIGURATION
# ============================================================

RETRIEVAL_TOP_K = 3


def get_pipeline_snapshot():
    """Return the configuration values that define a dataset release."""

    return {
        "random_seed": RANDOM_SEED,
        "sources": {
            "training": TRAIN_SOURCE,
            "test": TEST_SOURCE,
        },
        "split_sizes": {
            "training": TRAINING_SIZE,
            "validation": VALIDATION_SIZE,
            "test": TEST_SIZE,
        },
        "captions_per_image": CAPTIONS_PER_IMAGE,
        "models": {
            "clip": CLIP_MODEL_NAME,
            "sbert": SBERT_MODEL_NAME,
        },
        "embedding": {
            "clip_batch_size": CLIP_BATCH_SIZE,
            "sbert_batch_size": SBERT_BATCH_SIZE,
            "dtype": EMBEDDING_DTYPE,
            "normalize": NORMALIZE_EMBEDDINGS,
        },
        "retrieval": {
            "top_k": RETRIEVAL_TOP_K,
        },
    }


def load_saved_pipeline_snapshot(config_file=None):
    """Load the configuration snapshot saved with a dataset release."""

    config_file = config_file or (SPLITS_DIR / "split_config.json")

    if not config_file.is_file():
        raise FileNotFoundError(
            f"Pipeline configuration snapshot not found: {config_file}"
        )

    with open(config_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    snapshot = metadata.get("pipeline_configuration")

    if not isinstance(snapshot, dict):
        raise ValueError(
            f"Pipeline configuration snapshot is missing from {config_file}"
        )

    return snapshot


def check_pipeline_compatibility(
    saved_snapshot,
    artifact_name="artifact",
):
    """Raise an error when an artifact was created with another configuration."""

    active_snapshot = get_pipeline_snapshot()

    if saved_snapshot != active_snapshot:
        raise ValueError(
            f"{artifact_name} is incompatible with the active pipeline "
            "configuration.\n"
            f"Active configuration: {active_snapshot}\n"
            f"Saved configuration: {saved_snapshot}"
        )


def validate_saved_pipeline_compatibility(config_file=None):
    """Verify that the saved dataset release matches the active configuration."""

    snapshot = load_saved_pipeline_snapshot(config_file)
    check_pipeline_compatibility(snapshot, "Dataset release")