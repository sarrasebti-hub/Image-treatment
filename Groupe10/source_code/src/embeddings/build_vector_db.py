"""Build the FAISS vector database from precomputed CLIP embeddings.

Three indices are written under ``artifacts/vector_db/``:

* ``image_index.faiss``   one vector per Knowledge Base image
* ``caption_index.faiss`` one vector per Knowledge Base caption
* ``joint_index.faiss``   images + captions in one index (modality tag)

All vectors are L2-normalized, therefore FAISS inner product == cosine
similarity. The row order of each index is exactly the line order of the
matching JSONL metadata file.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import faiss
import numpy as np

from .config import (
    DEFAULT_MODEL,
    EMBEDDINGS_DIR,
    METADATA_DIR,
    VECTOR_DB_DIR,
    ensure_dirs,
    relative_to_project,
    resolve_project_path,
)
from .data import read_json, read_jsonl, write_json, write_jsonl


def load_embedding_artifacts(split: str) -> dict:
    manifest_path = EMBEDDINGS_DIR / f"{split}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run build_embeddings.py for '{split}' first."
        )

    manifest = read_json(manifest_path)
    files = manifest["files"]
    image_embeddings = np.load(resolve_project_path(files["images_embeddings"]))
    caption_embeddings = np.load(resolve_project_path(files["captions_embeddings"]))
    image_meta = read_jsonl(resolve_project_path(files["images_metadata"]))
    caption_meta = read_jsonl(resolve_project_path(files["captions_metadata"]))

    if image_embeddings.shape[0] != len(image_meta):
        raise ValueError("Image embeddings and image metadata have different lengths.")
    if caption_embeddings.shape[0] != len(caption_meta):
        raise ValueError("Caption embeddings and caption metadata have different lengths.")
    if image_embeddings.shape[1] != caption_embeddings.shape[1]:
        raise ValueError("Image and caption embedding dimensions differ.")

    return {
        "manifest": manifest,
        "image_embeddings": image_embeddings.astype(np.float32),
        "caption_embeddings": caption_embeddings.astype(np.float32),
        "image_meta": image_meta,
        "caption_meta": caption_meta,
    }


def make_index(
    vectors: np.ndarray,
    *,
    index_type: str = "flat",
    nlist: int | None = None,
    nprobe: int = 8,
):
    """Create and populate a FAISS inner-product index."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dimension = vectors.shape[1]

    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return index

    if index_type == "ivf":
        count = vectors.shape[0]
        if nlist is None:
            nlist = max(1, min(4096, int(round(np.sqrt(max(count, 1))))))
        if count < nlist:
            raise ValueError(
                f"IVF needs at least nlist={nlist} vectors, got {count}."
            )
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(vectors)
        index.add(vectors)
        index.nprobe = max(1, int(nprobe))
        return index

    raise ValueError(f"Unknown index type '{index_type}' (choose flat or ivf).")


def write_index(index, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(path)


def build_vector_db(
    *,
    split: str = "knowledge_base",
    index_type: str = "flat",
    nlist: int | None = None,
    nprobe: int = 8,
    force: bool = False,
) -> dict:
    """Build image, caption and joint FAISS indices for one split."""
    ensure_dirs()

    manifest_path = VECTOR_DB_DIR / "manifest.json"
    embeddings_manifest = EMBEDDINGS_DIR / f"{split}_manifest.json"
    if not force and manifest_path.exists():
        existing = read_json(manifest_path)
        database_is_newer = (
            embeddings_manifest.exists()
            and manifest_path.stat().st_mtime >= embeddings_manifest.stat().st_mtime
        )
        same_configuration = (
            existing.get("source_split") == split
            and existing.get("index_type") == index_type
        )
        if database_is_newer and same_configuration:
            print("Vector database already exists (use --force to rebuild), skipping.")
            return existing
        print("Embeddings or index configuration changed; rebuilding the vector database.")

    print(f"\n=== Building vector database from split '{split}' ===")
    artifacts = load_embedding_artifacts(split)
    image_embeddings = artifacts["image_embeddings"]
    caption_embeddings = artifacts["caption_embeddings"]
    image_meta = artifacts["image_meta"]
    caption_meta = artifacts["caption_meta"]

    dimension = int(image_embeddings.shape[1])
    model_name = artifacts["manifest"].get("model", {}).get("model_name", DEFAULT_MODEL)

    print(
        f"Indexing {len(image_meta)} images and {len(caption_meta)} captions "
        f"(dim={dimension}, type={index_type})..."
    )
    started = time.perf_counter()

    image_index = make_index(
        image_embeddings, index_type=index_type, nlist=nlist, nprobe=nprobe
    )
    caption_index = make_index(
        caption_embeddings, index_type=index_type, nlist=nlist, nprobe=nprobe
    )

    joint_vectors = np.vstack([image_embeddings, caption_embeddings]).astype(np.float32)
    joint_index = make_index(
        joint_vectors, index_type=index_type, nlist=nlist, nprobe=nprobe
    )

    joint_meta = [
        {**record, "modality": "image", "joint_id": position}
        for position, record in enumerate(image_meta)
    ]
    joint_meta.extend(
        {
            **record,
            "modality": "caption",
            "joint_id": len(image_meta) + position,
        }
        for position, record in enumerate(caption_meta)
    )

    image_path = VECTOR_DB_DIR / "image_index.faiss"
    caption_path = VECTOR_DB_DIR / "caption_index.faiss"
    joint_path = VECTOR_DB_DIR / "joint_index.faiss"
    joint_meta_path = METADATA_DIR / f"{split}_joint.jsonl"

    write_index(image_index, image_path)
    write_index(caption_index, caption_path)
    write_index(joint_index, joint_path)
    write_jsonl(joint_meta_path, joint_meta)

    build_time = time.perf_counter() - started

    manifest = {
        "task": "vector_database",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": split,
        "model": model_name,
        "embedding_dim": dimension,
        "metric": "cosine (inner product on L2-normalized vectors)",
        "index_type": index_type,
        "nlist": nlist if index_type == "ivf" else None,
        "nprobe": nprobe if index_type == "ivf" else None,
        "build_time_sec": round(build_time, 4),
        "row_order": "row i of an index == line i of its metadata JSONL file",
        "indices": {
            "images": {
                "path": relative_to_project(image_path),
                "metadata": relative_to_project(METADATA_DIR / f"{split}_images.jsonl"),
                "count": int(image_index.ntotal),
                "dimension": int(image_index.d),
            },
            "captions": {
                "path": relative_to_project(caption_path),
                "metadata": relative_to_project(METADATA_DIR / f"{split}_captions.jsonl"),
                "count": int(caption_index.ntotal),
                "dimension": int(caption_index.d),
            },
            "joint": {
                "path": relative_to_project(joint_path),
                "metadata": relative_to_project(joint_meta_path),
                "count": int(joint_index.ntotal),
                "dimension": int(joint_index.d),
            },
        },
    }
    write_json(manifest_path, manifest)

    print(
        f"Vector database ready: {image_index.ntotal} images, "
        f"{caption_index.ntotal} captions, {joint_index.ntotal} joint vectors "
        f"({build_time:.2f}s)."
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FAISS vector database.")
    parser.add_argument("--split", default="knowledge_base")
    parser.add_argument("--index-type", choices=["flat", "ivf"], default="flat")
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_vector_db(
        split=args.split,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
        force=args.force,
    )


if __name__ == "__main__":
    main()
