"""Encoders and a retrieval API for the Phase 1 knowledge-base embeddings.

Build the artifacts with ``python build_embeddings.py`` and check them with
``python check_embeddings.py``. Example::

    from retrieval import KnowledgeBase

    kb = KnowledgeBase()
    query = kb.encode_images([kb.image_path(3156)])             # a test image
    similar_images = kb.search_images(query, k=5)[0]           # [Hit(image_id, score), ...]
    captions = kb.search_captions(query, k=5, space="clip")[0]  # image -> KB captions

Validation images are also knowledge-base images. When a validation image is the
query, pass ``exclude=[{image_id}]`` so that neither the image nor its own
captions can be retrieved (anti-leakage rule).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Sequence

os.environ.setdefault("USE_TF", "0")  # PyTorch only: stop transformers from importing TensorFlow

# Import order matters on Windows: if torch is already loaded, importing transformers' model
# code AFTER faiss segfaults (exit 139). Loading the model classes before faiss is safe
# whatever the caller imported first, except "torch, then faiss, then this module".
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPModel, CLIPTokenizerFast  # noqa: E402

# isort: split
import faiss  # noqa: E402  (must stay after the transformers import above)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cleaned_dataset"
EMBED_DIR = ROOT / "embeddings"
CLIP_MAX_TOKENS = 77
SPACES = ("clip", "sbert")


@dataclass(frozen=True)
class EncoderSpec:
    """A Hugging Face encoder pinned to one commit, plus its output folder."""

    repo_id: str
    revision: str
    folder: str
    weights: str
    files: tuple[str, ...]


CLIP = EncoderSpec(
    repo_id="openai/clip-vit-base-patch32",
    revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    folder="clip_vit_b32",
    weights="pytorch_model.bin",
    files=("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
           "vocab.json", "merges.txt", "special_tokens_map.json", "pytorch_model.bin"),
)
SBERT = EncoderSpec(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    folder="minilm_l6_v2",
    weights="model.safetensors",
    files=("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
           "model.safetensors", "modules.json", "sentence_bert_config.json", "1_Pooling/config.json"),
)


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    """Write indented UTF-8 JSON with a trailing newline."""
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_caption(text: str) -> str:
    """Return the text that is embedded: the original caption with whitespace collapsed."""
    return " ".join(text.split())


def load_rgb_image(path: str | Path) -> Image.Image:
    """Open an image, apply its EXIF orientation and convert it to RGB."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def fetch_model(spec: EncoderSpec) -> Path:
    """Download (or reuse from the Hugging Face cache) the pinned files of an encoder.

    Loading from the local snapshot folder rather than the repo id also stops
    transformers from starting its background safetensors auto-conversion request
    for repositories that only ship ``pytorch_model.bin``.
    """
    return Path(snapshot_download(spec.repo_id, revision=spec.revision, allow_patterns=list(spec.files)))


def _unit(features: torch.Tensor) -> np.ndarray:
    """L2-normalise rows and return float32 NumPy, so inner product equals cosine."""
    return torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()


class Encoders:
    """CLIP image/text and Sentence-BERT text encoders, loaded on first use (inference only)."""

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._clip: tuple[CLIPModel, CLIPImageProcessor, CLIPTokenizerFast] | None = None
        self._sbert: tuple[Any, Any, int] | None = None

    def clip(self) -> tuple[CLIPModel, CLIPImageProcessor, CLIPTokenizerFast]:
        """Return the CLIP model, its reference (PIL-based) image processor and tokenizer."""
        if self._clip is None:
            path = fetch_model(CLIP)
            model = CLIPModel.from_pretrained(path).to(self.device).eval()
            self._clip = (model, CLIPImageProcessor.from_pretrained(path), CLIPTokenizerFast.from_pretrained(path))
        return self._clip

    def sbert(self) -> tuple[Any, Any, int]:
        """Return the MiniLM model, tokenizer and max sequence length (mean pooling + L2 norm)."""
        if self._sbert is None:
            path = fetch_model(SBERT)
            modules = read_json(path / "modules.json")
            pooling = read_json(path / "1_Pooling" / "config.json")
            if not pooling.get("pooling_mode_mean_tokens") or not any(m["type"].endswith("Normalize") for m in modules):
                raise ValueError(f"{SBERT.repo_id}: expected mean pooling followed by Normalize")
            max_length = int(read_json(path / "sentence_bert_config.json")["max_seq_length"])
            model = AutoModel.from_pretrained(path).to(self.device).eval()
            self._sbert = (model, AutoTokenizer.from_pretrained(path), max_length)
        return self._sbert

    @torch.inference_mode()
    def encode_images(self, paths: Sequence[str | Path], batch_size: int = 128, workers: int = 8) -> np.ndarray:
        """Embed images with CLIP: EXIF orientation, RGB, CLIP resize/crop/normalise, L2 norm."""
        if isinstance(paths, (str, Path)):
            raise TypeError("encode_images expects a list of paths: encode_images([path])")
        if not paths:
            raise ValueError("encode_images needs at least one image path")
        model, processor, _ = self.clip()

        def prepare(path: str | Path) -> np.ndarray:
            return processor(images=load_rgb_image(path), return_tensors="np")["pixel_values"][0]

        chunks = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(paths), batch_size):
                pixels = torch.from_numpy(np.stack(list(pool.map(prepare, paths[start:start + batch_size]))))
                chunks.append(_unit(model.get_image_features(pixel_values=pixels.to(self.device))))
                done = start + len(pixels)
                if len(paths) > batch_size and (done % (10 * batch_size) == 0 or done == len(paths)):
                    logger.info("Encoded %d/%d images", done, len(paths))
        return np.concatenate(chunks)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], space: str = "clip", batch_size: int = 512) -> np.ndarray:
        """Embed texts, after whitespace normalisation, in the CLIP or Sentence-BERT space."""
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {space!r}")
        if isinstance(texts, str):
            raise TypeError("encode_texts expects a list of texts: encode_texts([text])")
        if not texts:
            raise ValueError("encode_texts needs at least one text")
        texts = [normalize_caption(text) for text in texts]
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if space == "clip":
                model, _, tokenizer = self.clip()
                tokens = tokenizer(batch, padding=True, truncation=True, max_length=CLIP_MAX_TOKENS,
                                   return_tensors="pt").to(self.device)
                chunks.append(_unit(model.get_text_features(**tokens)))
            else:
                model, tokenizer, max_length = self.sbert()
                tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                                   return_tensors="pt").to(self.device)
                hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                chunks.append(_unit((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)))
        return np.concatenate(chunks)

    def token_counts(self, texts: Sequence[str], space: str = "clip") -> list[int]:
        """Untruncated token counts (special tokens included), used to report truncation."""
        if isinstance(texts, str):
            raise TypeError("token_counts expects a list of texts")
        tokenizer = self.clip()[2] if space == "clip" else self.sbert()[1]
        return [len(ids) for ids in tokenizer([normalize_caption(t) for t in texts])["input_ids"]]


@dataclass(frozen=True)
class Hit:
    """A retrieved knowledge-base item; caption fields are set only by caption searches."""

    image_id: int
    score: float
    caption_id: int | None = None
    caption: str | None = None


class KnowledgeBase:
    """Read-only access to the saved embeddings, FAISS indexes and row-to-ID maps."""

    def __init__(self, embed_dir: str | Path = EMBED_DIR, data_dir: str | Path = DATA_DIR,
                 device: str = "auto") -> None:
        self.embed_dir, self.data_dir = Path(embed_dir), Path(data_dir)
        self.encoders = Encoders(device)
        self.kb_image_ids = np.asarray(read_json(self.embed_dir / "kb_image_ids.json"), dtype=np.int64)
        self.test_image_ids = np.asarray(read_json(self.embed_dir / "test_image_ids.json"), dtype=np.int64)
        validation = read_json(self.embed_dir / "validation_kb_rows.json")
        self.validation_image_ids = np.asarray([r["image_id"] for r in validation], dtype=np.int64)
        self.validation_kb_rows = np.asarray([r["kb_row"] for r in validation], dtype=np.int64)
        with (self.embed_dir / "kb_captions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.caption_image_ids = np.asarray([int(r["image_id"]) for r in rows], dtype=np.int64)
        self.caption_ids = np.asarray([int(r["caption_id"]) for r in rows], dtype=np.int64)
        self.caption_texts = [r["caption_raw"] for r in rows]
        self._captions_per_image = Counter(self.caption_image_ids.tolist())
        self._kb_ids = set(self.kb_image_ids.tolist())
        self._test_ids = set(self.test_image_ids.tolist())
        self.image_index = faiss.read_index(str(self.embed_dir / CLIP.folder / "kb_image.faiss"))
        self.caption_indexes = {
            "clip": faiss.read_index(str(self.embed_dir / CLIP.folder / "kb_caption.faiss")),
            "sbert": faiss.read_index(str(self.embed_dir / SBERT.folder / "kb_caption.faiss")),
        }

    def image_path(self, image_id: int) -> Path:
        """Path of a knowledge-base (validation included) or test image file."""
        if image_id in self._kb_ids:
            split = "knowledge_base"
        elif image_id in self._test_ids:
            split = "test"
        else:
            raise KeyError(f"Image {image_id} is not in the knowledge base or test split")
        return self.data_dir / split / "images" / f"{image_id:012d}.jpg"

    def captions_for_image(self, image_id: int) -> list[str]:
        """Original captions of a knowledge-base image. Test captions are never stored here."""
        return [self.caption_texts[row] for row in np.flatnonzero(self.caption_image_ids == image_id)]

    def image_embeddings(self, split: str = "knowledge_base") -> np.ndarray:
        """Stored CLIP image vectors for 'knowledge_base', 'validation' (KB rows) or 'test'."""
        folder = self.embed_dir / CLIP.folder
        if split == "test":
            return np.load(folder / "test_image_embeddings.npy")
        if split not in ("knowledge_base", "validation"):
            raise ValueError(f"Unknown split {split!r}")
        kb = np.load(folder / "kb_image_embeddings.npy")
        return kb[self.validation_kb_rows] if split == "validation" else kb

    def caption_embeddings(self, space: str = "clip") -> np.ndarray:
        """Stored KB caption vectors in the 'clip' or 'sbert' space (row = kb_captions.csv row)."""
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {space!r}")
        return np.load(self.embed_dir / (CLIP if space == "clip" else SBERT).folder / "kb_caption_embeddings.npy")

    def encode_images(self, paths: Sequence[str | Path], **kwargs: Any) -> np.ndarray:
        """Embed query images exactly as the knowledge base was embedded."""
        return self.encoders.encode_images(paths, **kwargs)

    def encode_texts(self, texts: Sequence[str], space: str = "clip", **kwargs: Any) -> np.ndarray:
        """Embed query texts exactly as the knowledge-base captions were embedded."""
        return self.encoders.encode_texts(texts, space, **kwargs)

    def search_images(self, queries: np.ndarray, k: int = 5,
                      exclude: Sequence[Collection[int]] | None = None) -> list[list[Hit]]:
        """Nearest KB images for CLIP vectors (image or CLIP text queries), best first."""
        found = _search(self.image_index, queries, k, exclude, self.kb_image_ids,
                        lambda image_id: int(image_id in self._kb_ids))
        return [[Hit(int(self.kb_image_ids[row]), score) for row, score in hits] for hits in found]

    def search_captions(self, queries: np.ndarray, k: int = 5, space: str = "clip",
                        exclude: Sequence[Collection[int]] | None = None) -> list[list[Hit]]:
        """Nearest KB captions. 'clip' takes CLIP image or text vectors; 'sbert' takes Sentence-BERT vectors."""
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {space!r}")
        found = _search(self.caption_indexes[space], queries, k, exclude, self.caption_image_ids,
                        self._captions_per_image.__getitem__)
        return [[Hit(int(self.caption_image_ids[row]), score, int(self.caption_ids[row]), self.caption_texts[row])
                 for row, score in hits] for hits in found]


def _search(index: faiss.Index, queries: np.ndarray, k: int, exclude: Sequence[Collection[int]] | None,
            row_image_ids: np.ndarray, rows_of: Callable[[int], int]) -> list[list[tuple[int, float]]]:
    """Search, over-fetching enough rows so that excluded images still leave k results."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
    if queries.shape[1] != index.d:
        raise ValueError(f"Query dimension {queries.shape[1]} does not match index dimension {index.d}")
    if exclude is not None and any(isinstance(ids, (str, bytes, int, np.integer)) for ids in exclude):
        # exclude=["23686"] would otherwise become {2, 3, 6, 8} and silently keep the query image.
        raise TypeError("exclude needs one collection of image IDs per query, e.g. exclude=[{image_id}]")
    excluded = [set(map(int, ids)) for ids in exclude] if exclude is not None else [set() for _ in queries]
    if len(excluded) != len(queries):
        raise ValueError("exclude needs one collection of image IDs per query")
    extra = max((sum(rows_of(image_id) for image_id in ids) for ids in excluded), default=0)
    scores, rows = index.search(queries, min(k + extra, index.ntotal))
    return [[(int(row), float(score)) for row, score in zip(q_rows, q_scores, strict=True)
             if row >= 0 and int(row_image_ids[row]) not in ids][:k]
            for q_rows, q_scores, ids in zip(rows, scores, excluded, strict=True)]
