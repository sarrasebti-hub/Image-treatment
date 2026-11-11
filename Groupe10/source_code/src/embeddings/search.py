"""Similarity search against the FAISS vector database.

Examples
--------
Text -> image:
    python -m src.embeddings.search --mode text-to-image \
        --query "a man riding a motorcycle on a dirt road" --k 5

Image -> text:
    python -m src.embeddings.search --mode image-to-text \
        --image path/to/photo.jpg --k 5

Image -> image:
    python -m src.embeddings.search --mode image-to-image \
        --image-id 391895 --k 5

Text -> text:
    python -m src.embeddings.search --mode text-to-text \
        --query "rows of motorbikes in a city" --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from .config import (
    DEFAULT_MODEL,
    SPLITS,
    VECTOR_DB_DIR,
    resolve_project_path,
)
from .data import ensure_split_images, read_json, read_jsonl
from .encoder import ClipEncoder

MODES = ("text-to-image", "image-to-text", "image-to-image", "text-to-text")


class VectorSearch:
    """Load the FAISS indices lazily and run similarity searches."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        manifest_path = VECTOR_DB_DIR / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run build_vector_db.py first."
            )
        self.manifest = read_json(manifest_path)
        self.model_name = model_name or self.manifest.get("model", DEFAULT_MODEL)
        self.source_split = self.manifest["source_split"]
        self._encoder: ClipEncoder | None = None
        self._indices: dict[str, faiss.Index] = {}
        self._metadata: dict[str, list[dict]] = {}
        self._caption_lookup: dict[int, str] | None = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> ClipEncoder:
        if self._encoder is None:
            self._encoder = ClipEncoder(self.model_name)
        return self._encoder

    def _index(self, name: str):
        if name not in self._indices:
            entry = self.manifest["indices"][name]
            path = resolve_project_path(entry["path"])
            if not path.exists():
                raise FileNotFoundError(f"Missing FAISS index: {path}")
            self._indices[name] = faiss.read_index(str(path))
        return self._indices[name]

    def _meta(self, name: str) -> list[dict]:
        if name not in self._metadata:
            entry = self.manifest["indices"][name]
            path = resolve_project_path(entry["metadata"])
            self._metadata[name] = read_jsonl(path)
        return self._metadata[name]

    def _caption_for_image(self, image_id: int) -> str | None:
        if self._caption_lookup is None:
            captions_path = SPLITS[self.source_split]["captions"]
            if not captions_path.exists():
                self._caption_lookup = {}
            else:
                payload = read_json(captions_path)
                lookup: dict[int, str] = {}
                for annotation in payload.get("annotations", []):
                    lookup.setdefault(int(annotation["image_id"]), annotation["caption"])
                self._caption_lookup = lookup
        return self._caption_lookup.get(image_id)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_text(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query text is empty.")
        return self.encoder.encode_texts([query], progress=False)

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"Invalid image file: {path} ({error})") from error
        return self.encoder.encode_images([path], progress=False)

    def resolve_image_id(self, image_id: int, *, download: bool = False) -> Path:
        """Return the local path of a Knowledge Base image by COCO image ID."""
        records = self._meta("images")
        record = next(
            (item for item in records if int(item["image_id"]) == int(image_id)),
            None,
        )
        if record is None:
            raise KeyError(
                f"Image ID {image_id} is not part of the indexed split "
                f"'{self.source_split}'."
            )

        images_dir = SPLITS[self.source_split]["images"]
        path = images_dir / record["file_name"]
        if download:
            ensure_split_images(
                self.source_split,
                [{"id": int(image_id), "file_name": record["file_name"]}],
                download=True,
                workers=1,
                skip_missing=False,
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {path}. Run the search with --download "
                "to fetch it from COCO."
            )
        return path

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(
        self,
        query_vector: np.ndarray,
        index_name: str,
        metadata_name: str,
        k: int,
    ) -> list[dict]:
        index = self._index(index_name)
        metadata = self._meta(metadata_name)
        k = max(1, min(int(k), index.ntotal))
        scores, ids = index.search(np.ascontiguousarray(query_vector, dtype=np.float32), k)
        results = []
        for rank, (score, vector_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if vector_id < 0:
                continue
            record = dict(metadata[int(vector_id)])
            record["rank"] = rank
            record["score"] = float(score)
            results.append(record)
        return results

    def text_to_image(self, query: str, k: int = 5) -> list[dict]:
        results = self._search(self.encode_text(query), "images", "images", k)
        for result in results:
            result["caption_preview"] = self._caption_for_image(int(result["image_id"]))
        return results

    def image_to_text(self, image_path: str | Path, k: int = 5) -> list[dict]:
        return self._search(self.encode_image(image_path), "captions", "captions", k)

    def image_to_image(self, image_path: str | Path, k: int = 5) -> list[dict]:
        return self._search(self.encode_image(image_path), "images", "images", k)

    def text_to_text(self, query: str, k: int = 5) -> list[dict]:
        return self._search(self.encode_text(query), "captions", "captions", k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_results(mode: str, results: list[dict]) -> None:
    if not results:
        print("No results.")
        return

    if mode == "text-to-image":
        print(f"\n{'Rank':<5} {'Score':<8} {'Image ID':<12} {'File':<20} Caption preview")
        print("-" * 100)
        for item in results:
            preview = (item.get("caption_preview") or "")[:60]
            print(
                f"{item['rank']:<5} {item['score']:<8.4f} "
                f"{item['image_id']:<12} {item['file_name']:<20} {preview}"
            )
    elif mode == "image-to-text":
        print(f"\n{'Rank':<5} {'Score':<8} {'Image ID':<12} Caption")
        print("-" * 100)
        for item in results:
            print(
                f"{item['rank']:<5} {item['score']:<8.4f} "
                f"{item['image_id']:<12} {item['caption'][:70]}"
            )
    elif mode == "image-to-image":
        print(f"\n{'Rank':<5} {'Score':<8} {'Image ID':<12} File")
        print("-" * 60)
        for item in results:
            print(
                f"{item['rank']:<5} {item['score']:<8.4f} "
                f"{item['image_id']:<12} {item['file_name']}"
            )
    elif mode == "text-to-text":
        print(f"\n{'Rank':<5} {'Score':<8} {'Image ID':<12} Caption")
        print("-" * 100)
        for item in results:
            print(
                f"{item['rank']:<5} {item['score']:<8.4f} "
                f"{item['image_id']:<12} {item['caption'][:70]}"
            )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the image/caption vector database.")
    parser.add_argument("--mode", choices=MODES, default="text-to-image")
    parser.add_argument("--query", help="Text query (text-to-image / text-to-text).")
    parser.add_argument("--image", help="Path to a query image (image-to-* modes).")
    parser.add_argument(
        "--image-id",
        type=int,
        help="COCO image ID from the indexed split (image-to-* modes).",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.add_argument("--download", action="store_true", help="Download --image-id if missing.")
    parser.add_argument("--interactive", action="store_true", help="Interactive text-to-image loop.")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search = VectorSearch(args.model)

    if args.interactive:
        print("Interactive text -> image search. Type 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in {"quit", "exit", "q"}:
                break
            if not query:
                continue
            results = search.text_to_image(query, k=args.k)
            if args.json:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                print_results("text-to-image", results)
        return

    if args.mode in {"text-to-image", "text-to-text"}:
        if not args.query:
            raise SystemExit(f"--query is required for mode {args.mode}.")
        if args.mode == "text-to-image":
            results = search.text_to_image(args.query, k=args.k)
        else:
            results = search.text_to_text(args.query, k=args.k)
    else:
        if args.image:
            image_path = Path(args.image)
        elif args.image_id is not None:
            image_path = search.resolve_image_id(args.image_id, download=args.download)
        else:
            raise SystemExit(
                f"--image or --image-id is required for mode {args.mode}."
            )

        if args.mode == "image-to-text":
            results = search.image_to_text(image_path, k=args.k)
        else:
            results = search.image_to_image(image_path, k=args.k)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_results(args.mode, results)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
