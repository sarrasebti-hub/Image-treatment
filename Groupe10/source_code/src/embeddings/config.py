"""Project paths and shared constants.

All paths are resolved from the project root so the pipeline can be launched
from any working directory with ``python -m src.embeddings.<module>``.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
VECTOR_DB_DIR = ARTIFACTS_DIR / "vector_db"

REPORTS_DIR = PROJECT_ROOT / "reports"

# CLIP ViT-B/32: 512-dimensional joint image/text embedding space.
# It is small enough to run on CPU and already cached by the team.
DEFAULT_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
EMBEDDING_DIM = 512

SPLITS = {
    "knowledge_base": {
        "dir": PROCESSED_DIR / "knowledge_base",
        "images": PROCESSED_DIR / "knowledge_base" / "images",
        "captions": PROCESSED_DIR / "knowledge_base" / "captions.json",
        # KB images come from COCO train2017.
        "coco_source": "train2017",
    },
    "validation": {
        "dir": PROCESSED_DIR / "validation",
        "images": PROCESSED_DIR / "validation" / "images",
        "captions": PROCESSED_DIR / "validation" / "captions.json",
        "coco_source": "train2017",
    },
    "test": {
        "dir": PROCESSED_DIR / "test",
        "images": PROCESSED_DIR / "test" / "images",
        "captions": PROCESSED_DIR / "test" / "captions.json",
        # Test images come from COCO val2017.
        "coco_source": "val2017",
    },
}


def ensure_dirs() -> None:
    """Create the artifact/report directories if they do not exist."""
    for directory in (
        EMBEDDINGS_DIR,
        METADATA_DIR,
        VECTOR_DB_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def relative_to_project(path: Path) -> str:
    """Return a portable path relative to the project root."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a manifest path (relative to the project root)."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)
