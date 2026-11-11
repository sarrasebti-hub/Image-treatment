"""CLIP encoder: a shared 512-d embedding space for images and texts.

The encoder L2-normalizes every vector, so the inner product used by FAISS is
exactly the cosine similarity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .config import DEFAULT_BATCH_SIZE, DEFAULT_MODEL


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, keeping zero vectors at zero."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class ClipEncoder:
    """Small wrapper around ``transformers`` CLIP.

    The wrapper handles the transformers 4.x and 5.x output types
    (``Tensor`` vs ``BaseModelOutputWithPooling``).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.num_threads = num_threads
        self._model = None
        self._processor = None
        self._dim: int | None = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> "ClipEncoder":
        if self._model is not None:
            return self

        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self.num_threads:
            torch.set_num_threads(max(1, int(self.num_threads)))

        print(f"[encoder] loading {self.model_name} on {self.device}...", flush=True)
        self._model = (
            CLIPModel.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._dim = int(self._model.config.projection_dim)
        print(f"[encoder] ready (dim={self._dim})", flush=True)
        return self

    @property
    def dim(self) -> int:
        self.load()
        assert self._dim is not None
        return self._dim

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.dim,
            "device": self.device,
            "normalized": "l2",
            "similarity": "cosine (inner product on normalized vectors)",
            "batch_size": self.batch_size,
        }

    # ------------------------------------------------------------------
    # Output conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(output) -> np.ndarray:
        import torch

        if isinstance(output, torch.Tensor):
            tensor = output
        elif hasattr(output, "pooler_output") and output.pooler_output is not None:
            tensor = output.pooler_output
        elif hasattr(output, "image_embeds") and output.image_embeds is not None:
            tensor = output.image_embeds
        elif hasattr(output, "text_embeds") and output.text_embeds is not None:
            tensor = output.text_embeds
        elif isinstance(output, (tuple, list)):
            tensor = output[0]
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")

        return tensor.detach().cpu().float().numpy()

    @staticmethod
    def _chunks(items: Sequence, size: int):
        for start in range(0, len(items), size):
            yield start, items[start:start + size]

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_images(
        self,
        paths: Iterable[str | Path],
        *,
        batch_size: int | None = None,
        progress: bool = True,
    ) -> np.ndarray:
        import torch

        self.load()
        paths = [Path(path) for path in paths]
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)

        batch_size = max(1, int(batch_size or self.batch_size))
        batches: list[np.ndarray] = []

        for start, batch in self._chunks(paths, batch_size):
            images = []
            for path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))

            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                output = self._model.get_image_features(**inputs)

            batches.append(self._to_numpy(output))

            if progress:
                done = min(start + len(batch), len(paths))
                print(f"[encoder] images {done}/{len(paths)}", flush=True)

        embeddings = np.vstack(batches).astype(np.float32)
        return l2_normalize(embeddings)

    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        progress: bool = True,
    ) -> np.ndarray:
        import torch

        self.load()
        texts = [str(text).strip() for text in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        batch_size = max(1, int(batch_size or self.batch_size))
        batches: list[np.ndarray] = []

        for start, batch in self._chunks(texts, batch_size):
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                output = self._model.get_text_features(**inputs)

            batches.append(self._to_numpy(output))

            if progress:
                done = min(start + len(batch), len(texts))
                print(f"[encoder] texts {done}/{len(texts)}", flush=True)

        embeddings = np.vstack(batches).astype(np.float32)
        return l2_normalize(embeddings)
