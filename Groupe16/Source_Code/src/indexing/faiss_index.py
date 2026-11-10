from time import perf_counter

import faiss
import numpy as np

from src.embeddings.encode import Encoder, load_embeddings
from src.preprocessing.metadata import load_rgb
from src.utils.core import file_hash, normalize_vectors, path, read_json, stamp, write_json


def search(index, rows, query, top_k, query_image_id=None):
    if len(rows) != index.ntotal:
        raise ValueError("FAISS rows do not match index.ntotal")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be positive")
    query = normalize_vectors(np.asarray(query, dtype=np.float32).reshape(1, -1))
    if query.shape[1] != index.d:
        raise ValueError("Query embedding dimension differs from index")
    # Text rows may contain all five captions of the query image; exclude all of them.
    excluded = sum(row["image_id"] == query_image_id for row in rows) if query_image_id is not None else 0
    scores, indices = index.search(query, min(index.ntotal, top_k + excluded))
    return [{**rows[int(j)], "row": int(j), "cosine_similarity": float(score)} for score, j in zip(scores[0], indices[0])
            if j >= 0 and rows[int(j)]["image_id"] != query_image_id][:top_k]


def build_indexes(cfg):
    arrays, rows, embedding_manifest = load_embeddings(cfg)
    folder = path(cfg, "indexes")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {**stamp(cfg), "embedding_manifest_sha256": file_hash(path(cfg, "embeddings") / "manifest.json"),
                "model_revision": embedding_manifest["resolved_revision"], "indexes": {}}
    for kind, vectors in arrays.items():
        started = perf_counter()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        seconds = perf_counter() - started
        filename = folder / f"{kind}.index"
        faiss.write_index(index, str(filename))
        write_json(folder / f"{kind}_rows.json", rows[kind])
        manifest["indexes"][kind] = {"type": "IndexFlatIP", "dimension": index.d, "vectors": index.ntotal,
                                        "metric": "inner_product_on_L2_normalized_vectors = cosine", "build_seconds": seconds,
                                        "bytes_on_disk": filename.stat().st_size,
                                        "files_sha256": {name: file_hash(folder / name) for name in (f"{kind}.index", f"{kind}_rows.json")}}
    write_json(folder / "manifest.json", manifest)
    return manifest


def load_indexes(cfg):
    arrays, embedding_rows, embedding_manifest = load_embeddings(cfg)
    folder = path(cfg, "indexes")
    manifest = read_json(folder / "manifest.json")
    if manifest["embedding_manifest_sha256"] != file_hash(path(cfg, "embeddings") / "manifest.json"):
        raise ValueError("FAISS indexes stale; rebuild indexes")
    indexes, rows = {}, {}
    for kind in ("image", "text"):
        if any(file_hash(folder / name) != sha for name, sha in manifest["indexes"][kind]["files_sha256"].items()):
            raise ValueError("FAISS file/mapping hashes differ from manifest")
        indexes[kind] = faiss.read_index(str(folder / f"{kind}.index"))
        rows[kind] = read_json(folder / f"{kind}_rows.json")
        if rows[kind] != embedding_rows[kind] or indexes[kind].ntotal != len(rows[kind]) or indexes[kind].d != arrays[kind].shape[1]:
            raise ValueError("FAISS row/dimension mismatch")
        if indexes[kind].metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Unexpected FAISS metric")
        # Exact index permits reconstruction: validate association, not merely counts.
        restored = indexes[kind].reconstruct_n(0, indexes[kind].ntotal)
        if not np.array_equal(restored, arrays[kind]):
            raise ValueError("FAISS stored vectors differ from embedding rows")
    return indexes, rows, embedding_manifest


def smoke_test(cfg, data, image_id=None, split="validation", kind="image"):
    from src.preprocessing.metadata import load_metadata
    records = load_metadata(cfg, split)
    record = next((r for r in records if r["image_id"] == image_id), None) if image_id is not None else records[0]
    if record is None:
        raise ValueError("Query ID is not in the requested split")
    indexes, rows, manifest = load_indexes(cfg)
    if kind == "text" and not manifest["shared_image_text_space"]:
        raise ValueError("Image-to-text retrieval requires a shared encoder space; DINOv2 supports image-index queries only")
    # Always use the exact encoder revision used to build the KB.
    import copy
    encoder_cfg = copy.deepcopy(cfg)
    encoder_cfg["embedding"]["revision"] = manifest["resolved_revision"] or cfg["embedding"]["revision"]
    if not manifest["shared_image_text_space"]:
        encoder_cfg["embedding"]["fallback_text_revision"] = manifest["text_resolved_revision"] or cfg["embedding"]["fallback_text_revision"]
    encoder = Encoder(encoder_cfg)
    filename = data.image_path(record["coco_source"], record)
    if file_hash(filename) != record["file_sha256"]:
        raise ValueError("Query image changed since metadata creation")
    with load_rgb(filename) as rgb:
        query = encoder.images([rgb])
    started = perf_counter()
    hits = search(indexes[kind], rows[kind], query, cfg["retrieval"]["top_k"], record["image_id"])
    latency_ms = (perf_counter() - started) * 1000
    kb = {r["image_id"]: r for r in load_metadata(cfg, "knowledge_base")}
    for hit in hits:
        captions = kb[hit["image_id"]]["captions"]
        hit["captions"] = [c["caption_raw"] for c in captions if kind == "image" or c["caption_id"] == hit["caption_id"]]
    result = {**stamp(cfg), "query_image_id": record["image_id"], "query_split": split, "index": kind,
              "latency_ms": latency_ms, "results": hits, "purpose": "Preprocessing retrieval smoke test; no caption generation or quality metric"}
    write_json(path(cfg, "reports") / f"smoke_{kind}.json", result)
    return result
