"""Encode one or more splits into CLIP embeddings.

Outputs (under ``artifacts/``)
------------------------------
embeddings/<split>_images.npy      float32 (N, 512), L2-normalized
embeddings/<split>_captions.npy    float32 (M, 512), L2-normalized
metadata/<split>_images.jsonl      one record per image row
metadata/<split>_captions.jsonl    one record per caption row
embeddings/<split>_manifest.json   run configuration and file map
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    EMBEDDINGS_DIR,
    METADATA_DIR,
    SPLITS,
    ensure_dirs,
    relative_to_project,
)
from .data import (
    caption_records,
    ensure_split_images,
    filter_split_to_available,
    image_records,
    load_split,
    write_json,
    write_jsonl,
)
from .encoder import ClipEncoder


def save_npy(path: Path, array: np.ndarray) -> None:
    """Atomically save a NumPy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as file:
        np.save(file, array)
    os.replace(temporary, path)


def build_split_embeddings(
    split: str,
    *,
    limit: int | None = None,
    download: bool = False,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    skip_missing: bool = False,
    workers: int = 8,
) -> dict:
    """Encode all images and captions of one split and persist the artifacts."""
    ensure_dirs()

    images_npy = EMBEDDINGS_DIR / f"{split}_images.npy"
    captions_npy = EMBEDDINGS_DIR / f"{split}_captions.npy"
    images_jsonl = METADATA_DIR / f"{split}_images.jsonl"
    captions_jsonl = METADATA_DIR / f"{split}_captions.jsonl"
    manifest_path = EMBEDDINGS_DIR / f"{split}_manifest.json"

    if not force and all(
        path.exists()
        for path in (images_npy, captions_npy, images_jsonl, captions_jsonl, manifest_path)
    ):
        from .data import read_json

        existing = read_json(manifest_path)
        same_configuration = (
            existing.get("limit") == limit
            and existing.get("seed") == seed
            and existing.get("model", {}).get("model_name") == model_name
        )
        if same_configuration:
            print(f"[{split}] artifacts already exist (use --force to rebuild), skipping.")
            return existing
        print(
            f"[{split}] existing artifacts were built with a different "
            "limit/model/seed; rebuilding."
        )

    print(f"\n=== Encoding split '{split}' ===")
    split_data = load_split(split, limit=limit, seed=seed)
    print(
        f"[{split}] selected {len(split_data['images'])} images and "
        f"{len(split_data['annotations'])} captions."
    )

    availability = ensure_split_images(
        split,
        split_data["images"],
        download=download,
        workers=workers,
        skip_missing=skip_missing,
    )
    if len(availability["available_ids"]) < len(split_data["images"]):
        split_data = filter_split_to_available(
            split_data, availability["available_ids"]
        )
        print(
            f"[{split}] continuing with {len(split_data['images'])} "
            "available image(s)."
        )

    image_meta = image_records(split_data)
    caption_meta = caption_records(split_data)

    encoder = ClipEncoder(model_name=model_name, batch_size=batch_size)
    image_embeddings = encoder.encode_images(
        [record["path"] for record in image_meta]
    )
    caption_embeddings = encoder.encode_texts(
        [record["caption"] for record in caption_meta]
    )

    if image_embeddings.shape[0] != len(image_meta):
        raise RuntimeError(
            f"Image embedding count mismatch: {image_embeddings.shape[0]} "
            f"vs {len(image_meta)} metadata records."
        )
    if caption_embeddings.shape[0] != len(caption_meta):
        raise RuntimeError(
            f"Caption embedding count mismatch: {caption_embeddings.shape[0]} "
            f"vs {len(caption_meta)} metadata records."
        )

    save_npy(images_npy, image_embeddings)
    save_npy(captions_npy, caption_embeddings)
    write_jsonl(images_jsonl, image_meta)
    write_jsonl(captions_jsonl, caption_meta)

    manifest = {
        "task": "embeddings",
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": encoder.info(),
        "seed": seed,
        "limit": limit,
        "counts": {
            "images": len(image_meta),
            "captions": len(caption_meta),
            "requested_images": availability["total"],
            "downloaded_images": availability["downloaded"],
            "failed_images": availability["failed"],
        },
        "files": {
            "images_embeddings": relative_to_project(images_npy),
            "captions_embeddings": relative_to_project(captions_npy),
            "images_metadata": relative_to_project(images_jsonl),
            "captions_metadata": relative_to_project(captions_jsonl),
        },
        "captions_per_image": _caption_distribution(split_data),
        "source": {
            "captions_json": relative_to_project(SPLITS[split]["captions"]),
            "images_dir": relative_to_project(SPLITS[split]["images"]),
        },
    }
    write_json(manifest_path, manifest)

    print(
        f"[{split}] saved {image_embeddings.shape} image embeddings and "
        f"{caption_embeddings.shape} caption embeddings."
    )
    return manifest


def _caption_distribution(split_data: dict) -> dict:
    counts: dict[str, int] = {}
    for captions in split_data["captions_by_image"].values():
        key = str(len(captions))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CLIP image/text embeddings for one or more splits."
    )
    parser.add_argument(
        "--splits",
        default="knowledge_base",
        help="Comma-separated splits (knowledge_base,validation,test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images per split (default: all).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing COCO images (the repo currently stores LFS pointers).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Continue when some images cannot be downloaded.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild artifacts even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    for split in splits:
        if split not in SPLITS:
            raise SystemExit(f"Unknown split '{split}'. Choose from {sorted(SPLITS)}")

    for split in splits:
        build_split_embeddings(
            split,
            limit=args.limit,
            download=args.download,
            model_name=args.model,
            batch_size=args.batch_size,
            seed=args.seed,
            force=args.force,
            skip_missing=args.skip_missing,
            workers=args.workers,
        )

    print("\nEmbedding step complete.")


if __name__ == "__main__":
    main()
