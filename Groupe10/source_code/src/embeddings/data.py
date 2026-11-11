"""Split loading, image validation and COCO image download.

The repository stores the processed images as Git LFS pointer files (about
130 bytes each). This module detects that situation and can download the real
images from the official COCO URLs found in ``captions.json``.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .config import DEFAULT_SEED, SPLITS

LFS_MARKER = b"git-lfs.github.com/spec/v1"
USER_AGENT = "RAG-Image-Captioning/1.0 (dataset preparation)"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically write a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """Atomically write a JSON Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def is_lfs_pointer(path: str | Path) -> bool:
    """Return True when the file is a Git LFS pointer instead of an image."""
    path = Path(path)
    try:
        with open(path, "rb") as file:
            head = file.read(256)
    except OSError:
        return False
    return LFS_MARKER in head


def is_valid_image(path: str | Path) -> bool:
    """Check that a file can be decoded by Pillow."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 512:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def image_url(image: dict, coco_source: str) -> str:
    """Return the best available download URL for a COCO image."""
    for key in ("coco_url", "flickr_url"):
        value = image.get(key)
        if value:
            return str(value)
    return f"http://images.cocodataset.org/{coco_source}/{image['file_name']}"


def download_file(
    url: str,
    destination: str | Path,
    *,
    timeout: int = 60,
    retries: int = 3,
) -> None:
    """Download ``url`` to ``destination`` atomically, with retries."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with open(temporary, "wb") as file:
                    shutil.copyfileobj(response, file)
            if not is_valid_image(temporary):
                raise ValueError(f"Downloaded file is not a valid image: {url}")
            os.replace(temporary, destination)
            return
        except Exception as error:  # noqa: BLE001 - retry every network error
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Could not download {url}: {last_error}")


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------

def load_split(
    split: str,
    *,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Load one processed split and return images + captions.

    Parameters
    ----------
    split:
        ``knowledge_base``, ``validation`` or ``test``.
    limit:
        Optional number of images to keep (deterministic random sample).
    seed:
        Random seed used when ``limit`` is set.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Choose from {sorted(SPLITS)}")

    captions_path = SPLITS[split]["captions"]
    if not captions_path.exists():
        raise FileNotFoundError(
            f"Missing {captions_path}. Run the dataset preparation step first "
            "(src/audit/prepare_dataset.py)."
        )

    payload = read_json(captions_path)
    images: list[dict] = payload.get("images", [])
    annotations: list[dict] = payload.get("annotations", [])

    if not images:
        raise ValueError(f"No images found in {captions_path}")

    captions_by_image: dict[int, list[dict]] = {}
    for annotation in annotations:
        captions_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    # Keep only images that actually have captions.
    images = [image for image in images if captions_by_image.get(int(image["id"]))]
    images.sort(key=lambda item: int(item["id"]))

    if limit is not None and limit > 0 and limit < len(images):
        rng = random.Random(seed)
        selected_ids = {int(image["id"]) for image in rng.sample(images, limit)}
        images = [image for image in images if int(image["id"]) in selected_ids]

    selected_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in annotations
        if int(annotation["image_id"]) in selected_ids
    ]

    captions_by_image = {
        image_id: captions_by_image[image_id]
        for image_id in selected_ids
    }

    if len(images) != len(selected_ids):
        raise ValueError(f"Duplicate image IDs detected in split '{split}'")

    return {
        "split": split,
        "images": images,
        "annotations": annotations,
        "captions_by_image": captions_by_image,
        "limit": limit,
        "seed": seed,
    }


def image_records(split_data: dict) -> list[dict]:
    """Metadata records in the exact row order of the image embedding matrix."""
    split = split_data["split"]
    images_dir = SPLITS[split]["images"]
    records = []
    for vector_id, image in enumerate(split_data["images"]):
        records.append(
            {
                "vector_id": vector_id,
                "image_id": int(image["id"]),
                "file_name": image["file_name"],
                "path": str((images_dir / image["file_name"]).resolve()),
                "split": split,
                "width": image.get("width"),
                "height": image.get("height"),
                "num_captions": len(split_data["captions_by_image"][int(image["id"])]),
            }
        )
    return records


def caption_records(split_data: dict) -> list[dict]:
    """Metadata records in the exact row order of the caption embedding matrix."""
    split = split_data["split"]
    images_dir = SPLITS[split]["images"]
    records = []
    vector_id = 0
    for image in split_data["images"]:
        image_id = int(image["id"])
        for annotation in split_data["captions_by_image"][image_id]:
            records.append(
                {
                    "vector_id": vector_id,
                    "caption_id": int(annotation["id"]),
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "path": str((images_dir / image["file_name"]).resolve()),
                    "split": split,
                    "caption": str(annotation["caption"]).strip(),
                }
            )
            vector_id += 1
    return records


# ---------------------------------------------------------------------------
# Image availability / download
# ---------------------------------------------------------------------------

def ensure_split_images(
    split: str,
    images: list[dict],
    *,
    download: bool = False,
    workers: int = 8,
    timeout: int = 60,
    retries: int = 3,
    skip_missing: bool = False,
) -> dict:
    """Make sure every selected image is a real, decodable image file.

    Returns a report with ``available_ids`` (image IDs that can be encoded).
    Raises a clear error when images are missing and ``download`` is False.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'")

    images_dir = SPLITS[split]["images"]
    coco_source = SPLITS[split]["coco_source"]
    images_dir.mkdir(parents=True, exist_ok=True)

    present: list[dict] = []
    to_download: list[dict] = []
    for image in images:
        path = images_dir / image["file_name"]
        if is_valid_image(path):
            present.append(image)
        else:
            to_download.append(image)

    downloaded: list[dict] = []
    failed: list[dict] = []

    if to_download and download:
        print(
            f"[{split}] {len(present)} images already valid, "
            f"downloading {len(to_download)} missing image(s) from COCO..."
        )

        def worker(image: dict) -> dict:
            path = images_dir / image["file_name"]
            download_file(
                image_url(image, coco_source),
                path,
                timeout=timeout,
                retries=retries,
            )
            return image

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(worker, image): image for image in to_download}
            completed = 0
            for future in as_completed(futures):
                image = futures[future]
                completed += 1
                try:
                    future.result()
                    downloaded.append(image)
                except Exception as error:  # noqa: BLE001 - report and continue
                    failed.append({**image, "error": str(error)})
                if completed % 50 == 0 or completed == len(to_download):
                    print(
                        f"[{split}] downloaded {completed}/{len(to_download)} "
                        f"(failures: {len(failed)})",
                        flush=True,
                    )
    else:
        failed = [
            {**image, "error": "missing or Git LFS pointer"}
            for image in to_download
        ]

    available_ids = {
        int(image["id"]) for image in present
    } | {
        int(image["id"]) for image in downloaded
    }

    report = {
        "split": split,
        "total": len(images),
        "already_valid": len(present),
        "downloaded": len(downloaded),
        "failed": len(failed),
        "available_ids": available_ids,
        "failed_examples": failed[:10],
    }

    print(
        f"[{split}] images ready: {len(available_ids)}/{len(images)} "
        f"(valid={len(present)}, downloaded={len(downloaded)}, failed={len(failed)})"
    )

    if len(available_ids) < len(images):
        if not download:
            raise RuntimeError(
                f"{len(images) - len(available_ids)} image(s) of split '{split}' "
                "are missing or are Git LFS pointers. Re-run with --download to "
                "fetch the real COCO images, or place them manually in "
                f"{images_dir}."
            )
        if not skip_missing:
            examples = ", ".join(
                item["file_name"] for item in failed[:5]
            )
            raise RuntimeError(
                f"{len(failed)} image(s) of split '{split}' could not be "
                f"downloaded (examples: {examples}). Use --skip-missing to "
                "continue with the images that are available."
            )

    return report


def filter_split_to_available(split_data: dict, available_ids: set[int]) -> dict:
    """Drop images (and their captions) that could not be obtained."""
    images = [
        image for image in split_data["images"]
        if int(image["id"]) in available_ids
    ]
    kept_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation for annotation in split_data["annotations"]
        if int(annotation["image_id"]) in kept_ids
    ]
    captions_by_image = {
        image_id: split_data["captions_by_image"][image_id]
        for image_id in kept_ids
    }
    return {
        **split_data,
        "images": images,
        "annotations": annotations,
        "captions_by_image": captions_by_image,
    }
