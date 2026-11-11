"""Shared configuration and helpers for the Phase 1 data pipeline .

Paths can be overridden with environment variables:
    DATA_ROOT    folder containing train2017/, val2017/ and annotations_trainval2017/
    PROJECT_DIR  repository root (splits/ and report/ are written here)
"""
import collections
import hashlib
import json
import os
import re
from pathlib import Path

from PIL import Image, ImageOps

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/input/datasets/sabahesaraki/2017-2017"))
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "."))
ANN = DATA_ROOT / "annotations_trainval2017" / "annotations"
OUT_SPLITS = PROJECT_DIR / "splits"
OUT_REPORT = PROJECT_DIR / "report"

SEED = 42
N_KB, N_VAL = 10_000, 100
QUOTA = {"simple": 300, "medium": 400, "complex": 300}  # test = 1,000 images
N_CAPS = 5  # captions kept per image

# SHA-256 of the frozen v1.0 ID files (reproducibility check)
FROZEN_SHA256 = {
    "kb_ids.json": "71d0184e4c7b6cfb82c6003af4f0efe3bcf3cd28f3bf2942022fb064282a0af4",
    "val_ids.json": "0cf26ead61ba55690c52039b23eddf92cf2086c97383a6dd9ab33fc520ae9e3f",
    "test_ids.json": "cb24befe35a03ce734e5e8b1282b615eb74db3f1678ea3953c837528dd8f609f",
}

# Caption matching levels used by the caption-leakage check
LEVELS = {
    "exact": lambda s: s.strip(),
    "lowercase": lambda s: s.strip().lower(),
    "normalized": lambda s: " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split()),
}


def ensure_dirs():
    OUT_SPLITS.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)


def find_img_dir(name):
    p = DATA_ROOT / name
    return p / name if (p / name).is_dir() else p


def img_dirs():
    return {s: find_img_dir(s) for s in ["train2017", "val2017"]}


def check_image(path):
    """Image must open, decode and convert to RGB (EXIF orientation fixed)."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            ImageOps.exif_transpose(im).convert("RGB")
        return True, ""
    except Exception as e:  # noqa: BLE001 - every failure is logged, never silent
        return False, str(e)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path, indent=None):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
