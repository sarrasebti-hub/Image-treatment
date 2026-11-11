"""Generate the Phase 1 embeddings and FAISS indexes used for retrieval.

Reads the frozen split files in cleaned_dataset/ and writes embeddings/:

  clip_vit_b32/kb_image_embeddings.npy     10,000 x 512  CLIP image vectors (knowledge base)
  clip_vit_b32/test_image_embeddings.npy      100 x 512  CLIP image vectors (test queries only)
  clip_vit_b32/kb_caption_embeddings.npy   50,036 x 512  CLIP text vectors of KB captions
  minilm_l6_v2/kb_caption_embeddings.npy   50,036 x 384  Sentence-BERT vectors of KB captions
  */kb_*.faiss                             exact cosine indexes (IndexFlatIP on unit vectors)
  kb_image_ids.json, test_image_ids.json, validation_kb_rows.json, kb_captions.csv   row maps
  metadata.json                            models, revisions, settings, timings, checksums

Validation images are KB images, so their vectors are KB rows (validation_kb_rows.json).
Test captions are never read here: they are evaluation references only.

Usage: python build_embeddings.py [--device cuda|cpu] [--overwrite]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

import faiss
import numpy as np
import PIL
import torch
from PIL import Image

from retrieval import (
    CLIP,
    CLIP_MAX_TOKENS,
    DATA_DIR,
    EMBED_DIR,
    SBERT,
    Encoders,
    EncoderSpec,
    fetch_model,
    normalize_caption,
    read_json,
    sha256_file,
    write_json,
)

logger = logging.getLogger("build_embeddings")
RELEASE = "phase1-embeddings-v1"
INPUT_FILES = ("knowledge_base/image_ids.json", "knowledge_base/manifest.json", "knowledge_base/captions.json",
               "validation/image_ids.json", "test/image_ids.json", "test/manifest.json")


def parse_args() -> argparse.Namespace:
    """Command-line options; defaults reproduce the shared release."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=EMBED_DIR)
    parser.add_argument("--device", default="auto", help="auto (default), cuda or cpu")
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8, help="image decoding threads")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing release in --out-dir")
    return parser.parse_args()


@contextlib.contextmanager
def timed(timings: dict[str, float], key: str) -> Iterator[None]:
    """Record the wall-clock seconds of a block under timings[key]."""
    start = time.perf_counter()
    yield
    timings[key] = round(time.perf_counter() - start, 3)
    logger.info("%s: %.1fs", key, timings[key])


def load_splits(data_dir: Path) -> dict[str, Any]:
    """Read the frozen split IDs and KB captions, and check the invariants the release relies on."""
    kb, validation, test = (read_json(data_dir / name / "image_ids.json")
                            for name in ("knowledge_base", "validation", "test"))
    if len(set(kb)) != len(kb) or len(set(test)) != len(test):
        raise ValueError("Duplicate image IDs in a split")
    if set(kb) & set(test) or not set(validation) <= set(kb):
        raise ValueError("Test overlaps the knowledge base, or validation is not inside it")
    for split, ids in (("knowledge_base", kb), ("test", test)):
        names = {r["image_id"]: r["file_name"] for r in read_json(data_dir / split / "manifest.json")}
        if any(names.get(i) != f"{i:012d}.jpg" for i in ids):
            raise ValueError(f"{split}: manifest file names do not follow the COCO <id:012d>.jpg pattern")
    annotations = read_json(data_dir / "knowledge_base" / "captions.json")["annotations"]
    if {a["image_id"] for a in annotations} != set(kb):
        raise ValueError("KB captions do not cover exactly the knowledge-base images")
    return {"kb": kb, "validation": validation, "test": test, "annotations": annotations}


def caption_rows(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per KB caption ordered by (image_id, caption_id); row i is embedding row i."""
    ordered = sorted(annotations, key=lambda a: (a["image_id"], a["id"]))
    return [{"row": i, "image_id": a["image_id"], "caption_id": a["id"], "caption_raw": a["caption"],
             "caption_embedded": normalize_caption(a["caption"])} for i, a in enumerate(ordered)]


def count_exif_rotations(paths: list[Path]) -> int:
    """Number of images whose EXIF orientation tag requires a transpose."""
    count = 0
    for path in paths:
        with Image.open(path) as image:
            count += image.getexif().get(0x0112, 1) != 1
    return count


def describe(path: Path, root: Path) -> dict[str, Any]:
    """Relative path, size and checksum of a written file."""
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def save_matrix(path: Path, matrix: np.ndarray, root: Path) -> dict[str, Any]:
    """Save a float32 matrix as .npy and describe it for metadata.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(matrix, dtype=np.float32))
    return {**describe(path, root), "shape": list(matrix.shape), "dtype": "float32"}


def save_index(path: Path, matrix: np.ndarray, root: Path) -> dict[str, Any]:
    """Build an exact inner-product FAISS index (cosine on unit vectors) and save it."""
    start = time.perf_counter()
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(np.ascontiguousarray(matrix, dtype=np.float32))
    seconds = time.perf_counter() - start
    faiss.write_index(index, str(path))
    return {**describe(path, root), "type": "faiss.IndexFlatIP (exact search)",
            "metric": "inner product on L2-normalised vectors = cosine similarity",
            "ntotal": int(index.ntotal), "dim": int(index.d), "build_seconds": round(seconds, 4)}


def encoder_record(spec: EncoderSpec, dim: int, extra: dict[str, Any]) -> dict[str, Any]:
    """Pinned model identity plus the settings used to embed with it."""
    return {"repo_id": spec.repo_id, "revision": spec.revision, "weights_file": spec.weights,
            "weights_sha256": sha256_file(fetch_model(spec) / spec.weights), "embedding_dim": dim,
            "dtype": "float32", "normalisation": "L2 (unit norm)", **extra}


def environment(device: torch.device) -> dict[str, Any]:
    """Library versions and hardware used for the build."""
    return {"python": platform.python_version(), "platform": platform.platform(), "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": version("transformers"),
            "huggingface_hub": version("huggingface_hub"), "faiss": faiss.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_allowed": torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32}


def write_caption_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write kb_captions.csv: the text and IDs behind every caption-index row."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Embed the knowledge base and test images, save vectors, indexes, row maps and metadata."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out_dir
    if (out / "metadata.json").exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite the release in {out}; pass --overwrite to rebuild it.")
    for stale in (out / "metadata.json", out / "qc_report.json"):
        stale.unlink(missing_ok=True)  # an interrupted rebuild must not keep the old "passed" report
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    splits = load_splits(args.data_dir)
    rows = caption_rows(splits["annotations"])
    texts = [row["caption_raw"] for row in rows]
    kb_paths = [args.data_dir / "knowledge_base" / "images" / f"{i:012d}.jpg" for i in splits["kb"]]
    test_paths = [args.data_dir / "test" / "images" / f"{i:012d}.jpg" for i in splits["test"]]
    logger.info("KB images %d, validation (KB rows) %d, test images %d, KB captions %d",
                len(kb_paths), len(splits["validation"]), len(test_paths), len(rows))

    encoders = Encoders(args.device)
    timings: dict[str, float] = {}
    with timed(timings, "download_and_load_models"):
        clip_model, processor, _ = encoders.clip()
        encoders.sbert()
    logger.info("Device: %s", encoders.device)
    with timed(timings, "encode_kb_images"):
        kb_images = encoders.encode_images(kb_paths, args.image_batch_size, args.workers)
    with timed(timings, "encode_test_images"):
        test_images = encoders.encode_images(test_paths, args.image_batch_size, args.workers)
    with timed(timings, "encode_kb_captions_clip"):
        clip_captions = encoders.encode_texts(texts, "clip", args.text_batch_size)
    with timed(timings, "encode_kb_captions_sbert"):
        sbert_captions = encoders.encode_texts(texts, "sbert", args.text_batch_size)

    clip_dir, sbert_dir = out / CLIP.folder, out / SBERT.folder
    artifacts = [save_matrix(clip_dir / "kb_image_embeddings.npy", kb_images, out),
                 save_matrix(clip_dir / "test_image_embeddings.npy", test_images, out),
                 save_matrix(clip_dir / "kb_caption_embeddings.npy", clip_captions, out),
                 save_matrix(sbert_dir / "kb_caption_embeddings.npy", sbert_captions, out)]
    indexes = [save_index(clip_dir / "kb_image.faiss", kb_images, out),
               save_index(clip_dir / "kb_caption.faiss", clip_captions, out),
               save_index(sbert_dir / "kb_caption.faiss", sbert_captions, out)]

    kb_row = {image_id: row for row, image_id in enumerate(splits["kb"])}
    write_json(out / "kb_image_ids.json", splits["kb"])
    write_json(out / "test_image_ids.json", splits["test"])
    write_json(out / "validation_kb_rows.json",
               [{"image_id": i, "kb_row": kb_row[i]} for i in sorted(splits["validation"])])
    write_caption_table(out / "kb_captions.csv", rows)

    clip_tokens, sbert_tokens = encoders.token_counts(texts, "clip"), encoders.token_counts(texts, "sbert")
    sbert_max_length = encoders.sbert()[2]
    metadata = {
        "release": RELEASE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["python", Path(sys.argv[0]).name, *sys.argv[1:]]),
        "dataset": {"kb_images": len(kb_paths), "validation_images": len(splits["validation"]),
                    "validation_note": "validation vectors are KB rows listed in validation_kb_rows.json",
                    "test_images": len(test_paths), "kb_captions": len(rows),
                    "input_sha256": {f"cleaned_dataset/{name}": sha256_file(args.data_dir / name)
                                     for name in INPUT_FILES}},
        "preprocessing": {
            "image_steps": ["PIL ImageOps.exif_transpose (EXIF orientation)", "convert to RGB",
                            "CLIPImageProcessor: resize shortest side, center crop, rescale 1/255, "
                            "normalise with CLIP mean/std (values below)"],
            "image_processor": {"size": processor.size, "crop_size": processor.crop_size,
                                "resample": "bicubic" if processor.resample == 3 else processor.resample,
                                "image_mean": processor.image_mean, "image_std": processor.image_std},
            "images_with_exif_rotation": count_exif_rotations(kb_paths + test_paths),
            "caption_text": "original caption with whitespace collapsed; case and punctuation kept",
            "captions_changed_by_whitespace_normalisation": sum(r["caption_raw"] != r["caption_embedded"]
                                                                for r in rows),
            "test_captions": "not read and not embedded (evaluation references only)",
        },
        "encoders": {
            "clip": encoder_record(CLIP, int(clip_model.config.projection_dim), {
                "used_for": ["KB images", "test images", "KB captions (shared image-text space)"],
                "image_batch_size": args.image_batch_size, "text_batch_size": args.text_batch_size,
                "max_text_tokens": CLIP_MAX_TOKENS, "max_caption_tokens": max(clip_tokens),
                "captions_truncated": sum(n > CLIP_MAX_TOKENS for n in clip_tokens)}),
            "sbert": encoder_record(SBERT, int(sbert_captions.shape[1]), {
                "used_for": ["KB captions (text-to-text space)"], "pooling": "attention-masked mean, then L2",
                "text_batch_size": args.text_batch_size, "max_text_tokens": sbert_max_length,
                "max_caption_tokens": max(sbert_tokens),
                "captions_truncated": sum(n > sbert_max_length for n in sbert_tokens)}),
        },
        "artifacts": artifacts,
        "indexes": indexes,
        "timings_seconds": timings,
        "environment": environment(encoders.device),
    }
    write_json(out / "metadata.json", metadata)
    logger.info("Wrote %s release to %s", RELEASE, out)


if __name__ == "__main__":
    main()
