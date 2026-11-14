from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_common import EXPERIMENT_ROOT, read_json


ENCODERS = {
    "clip-b32": ("clip", "openai/clip-vit-base-patch32", 512),
    "siglip-b16": ("siglip", "google/siglip-base-patch16-224", 768),
    "dinov2-s": ("dinov2", "facebook/dinov2-small", 384),
}


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encoder produced a zero embedding")
    return (values / norms).astype(np.float32)


def features(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "float"):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    return output.last_hidden_state[:, 0]


def encode_images(kind: str, model_name: str, paths, batch: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if kind in ("clip", "siglip"):
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)

    projection = getattr(model, "visual_projection", None) if kind == "clip" else None
    projection_dim = getattr(model.config, "projection_dim", None) if kind == "clip" else None
    output = []
    for start in range(0, len(paths), batch):
        images = [Image.open(path).convert("RGB") for path in paths[start:start + batch]]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            if kind in ("clip", "siglip"):
                encoded = model.get_image_features(pixel_values=pixel_values)
                encoded = features(encoded)
                if projection is not None and projection_dim is not None and encoded.shape[-1] != projection_dim:
                    encoded = projection(encoded)
            else:
                encoded = model(pixel_values=pixel_values).last_hidden_state[:, 0]
        output.append(encoded.float().cpu().numpy())
        del images
    vectors = np.concatenate(output, axis=0)
    return l2_normalize(vectors.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=list(ENCODERS), action="append")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    names = args.encoder or list(ENCODERS)

    split = read_json(EXPERIMENT_ROOT / "splits" / "test_1000.json")
    image_dir = EXPERIMENT_ROOT / "images_rgb" / "test_1000"
    ids = np.array([int(row["image_id"]) for row in split["images"]], dtype=np.int64)
    paths = [image_dir / f"{image_id:012d}.jpg" for image_id in ids]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} preprocessed images; first: {missing[0]}")

    for name in names:
        kind, model_name, expected_dim = ENCODERS[name]
        vectors = encode_images(kind, model_name, paths, args.batch)
        if vectors.shape != (len(ids), expected_dim):
            raise ValueError(f"{name}: expected {(len(ids), expected_dim)}, got {vectors.shape}")
        output = EXPERIMENT_ROOT / "embeddings" / name
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "ids_test_1000.npy", ids)
        np.save(output / "test_1000_img.npy", vectors)
        print(f"Saved {name} embeddings: shape={vectors.shape}")


if __name__ == "__main__":
    main()
