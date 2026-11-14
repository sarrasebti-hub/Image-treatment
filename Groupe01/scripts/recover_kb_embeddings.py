from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import load_config
ENCODERS = {
    "clip-b32": ("clip", "openai/clip-vit-base-patch32", 512),
    "siglip-b16": ("siglip", "google/siglip-base-patch16-224", 768),
    "dinov2-s": ("dinov2", "facebook/dinov2-small", 384),
}
TEXT_ENCODERS = {
    "clip-b32": "openai/clip-vit-base-patch32",
    "siglip-b16": "google/siglip-base-patch16-224",
}


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encoder produced a zero embedding")
    return (values / np.maximum(norms, np.finfo(np.float32).eps)).astype(np.float32)


def features(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    return output


def atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_train_source(requested: Path | None) -> Path:
    candidates = [
        requested,
        ROOT / "data" / "raw" / "images" / "train2017",
        Path(r"C:\Users\admin\Desktop\archive\coco2017\train2017"),
        Path(r"C:\Users\admin\Downloads\archive\coco2017\train2017"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    raise FileNotFoundError("COCO train2017 source directory was not found")


def restore_normalized_images(ids: np.ndarray, source: Path, destination: Path) -> list[Path]:
    cfg = load_config()
    max_side = int(cfg["image"]["max_side"])
    jpeg_quality = int(cfg["image"]["jpeg_quality"])
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for image_id in ids:
        destination_path = destination / f"{int(image_id):012d}.jpg"
        paths.append(destination_path)
        if destination_path.exists():
            continue
        source_path = source / f"{int(image_id):012d}.jpg"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = rgb.size
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            rgb = rgb.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        rgb.save(destination_path, format="JPEG", quality=jpeg_quality, subsampling=0)
    return paths


def encode_images(kind: str, model_name: str, paths: list[Path], batch: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if kind in ("clip", "siglip"):
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)

    all_vectors = []
    for start in range(0, len(paths), batch):
        batch_paths = paths[start:start + batch]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            if kind in ("clip", "siglip"):
                vectors = model.get_image_features(pixel_values=pixel_values)
                vectors = features(vectors)
            else:
                vectors = model(pixel_values=pixel_values).last_hidden_state[:, 0]
        all_vectors.append(vectors.float().cpu().numpy().astype(np.float32))
    return l2_normalize(np.concatenate(all_vectors, axis=0))


def encode_text(model_name: str, texts: list[str], batch: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    all_vectors = []
    for start in range(0, len(texts), batch):
        batch_texts = texts[start:start + batch]
        inputs = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                vectors = model.get_text_features(**inputs)
            else:
                vectors = model(**inputs).text_embeds
        vectors = features(vectors)
        all_vectors.append(vectors.float().cpu().numpy().astype(np.float32))
    return l2_normalize(np.concatenate(all_vectors, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--encoder", choices=list(ENCODERS), action="append")
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--text-batch", type=int, default=64)
    args = parser.parse_args()

    split = json.loads((ROOT / "artifacts" / "splits" / "kb.json").read_text(encoding="utf-8"))
    if isinstance(split, dict):
        rows = split["images"]
    else:
        rows = split
    ids = np.array([int(row["image_id"]) for row in rows], dtype=np.int64)
    if len(ids) != 10000 or len(np.unique(ids)) != 10000 or ids.tolist() != sorted(ids.tolist()):
        raise ValueError("Frozen KB split must contain 10000 unique sorted image IDs")

    source = find_train_source(args.train_dir)
    image_destination = ROOT / "artifacts" / "images_rgb" / "kb"
    image_paths = restore_normalized_images(ids, source, image_destination)

    captions = read_jsonl(ROOT / "artifacts" / "tables" / "kb_captions.jsonl")
    if len(captions) != 50000:
        raise ValueError(f"Expected 50000 KB captions, found {len(captions)}")
    names = args.encoder or list(ENCODERS)
    for name in names:
        kind, model_name, expected_dimension = ENCODERS[name]
        output = ROOT / "artifacts" / "embeddings" / name
        output.mkdir(parents=True, exist_ok=True)

        vectors = encode_images(kind, model_name, image_paths, args.image_batch)
        if vectors.shape != (len(ids), expected_dimension):
            raise ValueError(f"{name}: expected {(len(ids), expected_dimension)}, got {vectors.shape}")
        atomic_save(output / "ids_kb.npy", ids)
        atomic_save(output / "kb_img.npy", vectors)
        print(f"Recovered {name} image embeddings: {vectors.shape}")

        if name in TEXT_ENCODERS:
            caption_ids = np.array([int(row["caption_id"]) for row in captions], dtype=np.int64)
            image_ids = np.array([int(row["image_id"]) for row in captions], dtype=np.int64)
            text_vectors = encode_text(TEXT_ENCODERS[name], [str(row["caption_raw"]) for row in captions], args.text_batch)
            if text_vectors.shape != (len(captions), expected_dimension):
                raise ValueError(f"{name}: expected {(len(captions), expected_dimension)}, got {text_vectors.shape}")
            atomic_save(output / "kb_txt.npy", text_vectors)
            atomic_save(output / "kb_txt_caption_ids.npy", caption_ids)
            atomic_save(output / "kb_txt_image_ids.npy", image_ids)
            print(f"Recovered {name} text embeddings: {text_vectors.shape}")


if __name__ == "__main__":
    main()
