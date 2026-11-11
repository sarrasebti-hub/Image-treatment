"""Quality control for the embeddings and the FAISS vector database.

The report covers:

1. Embedding integrity: shape, NaNs/Infs, L2 norms, zero vectors, duplicates.
2. Metadata integrity: coverage, unique IDs, captions per image, caption stats.
3. Vector database integrity: counts, dimension, self-retrieval, file checksums.
4. Retrieval quality: text -> image and image -> text (Recall@K, MRR, ranks).
5. Performance: query encoding and FAISS search latency.
6. A PASS / PASS WITH WARNINGS / FAIL verdict with recommendations.

Outputs:
    reports/qc_report.md
    reports/qc_metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from .build_vector_db import load_embedding_artifacts
from .config import (
    DEFAULT_MODEL,
    DEFAULT_SEED,
    REPORTS_DIR,
    VECTOR_DB_DIR,
    ensure_dirs,
    relative_to_project,
    resolve_project_path,
)
from .data import read_json, read_jsonl
from .encoder import ClipEncoder


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hashes(embeddings: np.ndarray) -> set[str]:
    return {hashlib.sha256(row.tobytes()).hexdigest() for row in embeddings}


def _sample_indices(count: int, limit: int, seed: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    rng = random.Random(seed)
    return sorted(rng.sample(range(count), limit))


def _round(value, digits: int = 4):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Embedding checks
# ---------------------------------------------------------------------------

def check_embeddings(
    name: str,
    embeddings: np.ndarray,
    metadata: list[dict],
    *,
    id_key: str,
    near_duplicate_threshold: float = 0.999,
    sample_size: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    count, dimension = embeddings.shape

    finite = bool(np.isfinite(embeddings).all())
    norms = np.linalg.norm(embeddings, axis=1) if count else np.zeros(0)
    norm_deviation = (
        float(np.max(np.abs(norms - 1.0))) if count else None
    )

    exact_duplicate_rows = count - len(_row_hashes(embeddings)) if count else 0

    # Near duplicates on a bounded sample (cosine similarity).
    sample = _sample_indices(count, sample_size, seed)
    near_duplicate_pairs = 0
    max_off_diagonal = None
    if len(sample) > 1:
        sample_vectors = embeddings[sample]
        similarity = sample_vectors @ sample_vectors.T
        np.fill_diagonal(similarity, -np.inf)
        upper = similarity[np.triu_indices(len(sample), k=1)]
        near_duplicate_pairs = int((upper >= near_duplicate_threshold).sum())
        max_off_diagonal = float(upper.max()) if upper.size else None

    ids = [record.get(id_key) for record in metadata]
    unique_ids = len(set(ids))
    vector_ids = [record.get("vector_id") for record in metadata]
    vector_ids_ok = vector_ids == list(range(len(metadata)))

    caption_stats = None
    if name == "captions":
        lengths = [len(str(record.get("caption", "")).split()) for record in metadata]
        empty = sum(
            1 for record in metadata
            if not str(record.get("caption", "")).strip()
        )
        caption_stats = {
            "empty_captions": empty,
            "words_min": int(min(lengths)) if lengths else 0,
            "words_max": int(max(lengths)) if lengths else 0,
            "words_mean": _round(np.mean(lengths)) if lengths else 0,
            "words_median": _round(np.median(lengths)) if lengths else 0,
        }

    return {
        "name": name,
        "count": int(count),
        "dimension": int(dimension),
        "metadata_count": len(metadata),
        "metadata_matches": int(count) == len(metadata),
        "finite": finite,
        "nan_count": int(np.isnan(embeddings).sum()),
        "inf_count": int(np.isinf(embeddings).sum()),
        "norm_mean": _round(np.mean(norms)) if count else None,
        "norm_std": _round(np.std(norms)) if count else None,
        "norm_min": _round(np.min(norms)) if count else None,
        "norm_max": _round(np.max(norms)) if count else None,
        "unit_norm_max_deviation": _round(norm_deviation),
        "zero_vectors": int((norms < 1e-8).sum()) if count else 0,
        "exact_duplicate_rows": int(exact_duplicate_rows),
        "near_duplicate_pairs": near_duplicate_pairs,
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_sample_size": len(sample),
        "max_off_diagonal_similarity": _round(max_off_diagonal),
        "unique_ids": unique_ids,
        "unique_ids_ok": unique_ids == len(metadata),
        "vector_ids_sequential": vector_ids_ok,
        "caption_stats": caption_stats,
    }


# ---------------------------------------------------------------------------
# Metadata checks
# ---------------------------------------------------------------------------

def check_metadata(
    split: str,
    image_meta: list[dict],
    caption_meta: list[dict],
    *,
    check_files: bool = True,
) -> dict:
    image_ids = [int(record["image_id"]) for record in image_meta]
    caption_image_ids = [int(record["image_id"]) for record in caption_meta]

    image_id_set = set(image_ids)
    caption_image_id_set = set(caption_image_ids)

    captions_per_image: dict[int, int] = {}
    for image_id in caption_image_ids:
        captions_per_image[image_id] = captions_per_image.get(image_id, 0) + 1

    distribution: dict[str, int] = {}
    for count in captions_per_image.values():
        key = str(count)
        distribution[key] = distribution.get(key, 0) + 1

    normalized_captions = [
        " ".join(str(record.get("caption", "")).lower().split())
        for record in caption_meta
    ]
    duplicate_captions = len(normalized_captions) - len(set(normalized_captions))

    missing_files = 0
    lfs_pointer_files = 0
    if check_files:
        for record in image_meta:
            path = Path(record.get("path", ""))
            if not path.is_file():
                missing_files += 1
                continue
            try:
                with open(path, "rb") as file:
                    head = file.read(256)
                if b"git-lfs.github.com/spec/v1" in head:
                    lfs_pointer_files += 1
            except OSError:
                missing_files += 1

    return {
        "split": split,
        "images": len(image_meta),
        "captions": len(caption_meta),
        "unique_image_ids": len(image_id_set),
        "unique_caption_image_ids": len(caption_image_id_set),
        "captions_referencing_unknown_images": len(caption_image_id_set - image_id_set),
        "images_without_captions": len(image_id_set - caption_image_id_set),
        "captions_per_image_distribution": dict(
            sorted(distribution.items(), key=lambda item: int(item[0]))
        ),
        "average_captions_per_image": _round(
            len(caption_meta) / len(image_meta) if image_meta else 0
        ),
        "duplicate_caption_texts": int(duplicate_captions),
        "duplicate_caption_rate": _round(
            duplicate_captions / len(caption_meta) if caption_meta else 0
        ),
        "missing_image_files": missing_files,
        "lfs_pointer_files": lfs_pointer_files,
    }


# ---------------------------------------------------------------------------
# FAISS checks
# ---------------------------------------------------------------------------

def check_index(
    name: str,
    index_path: Path,
    metadata: list[dict],
    embeddings: np.ndarray,
    *,
    expected_dimension: int,
    sample_size: int = 100,
    seed: int = DEFAULT_SEED,
) -> dict:
    index = faiss.read_index(str(index_path))
    sample = _sample_indices(int(embeddings.shape[0]), sample_size, seed)

    self_retrieval_accuracy = None
    reconstruction_error = None

    if sample:
        query = np.ascontiguousarray(embeddings[sample], dtype=np.float32)
        _, ids = index.search(query, 1)
        correct = sum(
            1 for row, expected in enumerate(sample)
            if int(ids[row][0]) == int(expected)
        )
        self_retrieval_accuracy = correct / len(sample)

    # Exact reconstruction is only guaranteed by a flat index.
    if index.__class__.__name__ == "IndexFlatIP" and sample:
        try:
            reconstructed = np.vstack(
                [index.reconstruct(int(position)) for position in sample]
            )
            reconstruction_error = float(
                np.max(np.abs(reconstructed - embeddings[sample]))
            )
        except Exception:  # noqa: BLE001 - not critical for the verdict
            reconstruction_error = None

    return {
        "name": name,
        "path": relative_to_project(index_path),
        "count": int(index.ntotal),
        "metadata_count": len(metadata),
        "count_matches": int(index.ntotal) == len(metadata),
        "dimension": int(index.d),
        "expected_dimension": int(expected_dimension),
        "dimension_matches": int(index.d) == int(expected_dimension),
        "index_type": index.__class__.__name__,
        "file_size_bytes": index_path.stat().st_size,
        "sha256": sha256_file(index_path),
        "self_retrieval_accuracy": _round(self_retrieval_accuracy),
        "reconstruction_max_abs_error": _round(reconstruction_error, 6),
    }


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def _flat_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _summarize_ranks(ranks: list[int | None], ks: list[int]) -> dict:
    total = len(ranks)
    if total == 0:
        return {"n_queries": 0}

    summary = {"n_queries": total}
    for k in ks:
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)
        summary[f"recall@{k}"] = _round(hits / total)

    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
    summary["mrr"] = _round(float(np.mean(reciprocal)))

    found = [rank for rank in ranks if rank is not None]
    summary["found_rate"] = _round(len(found) / total)
    summary["median_rank_found"] = _round(np.median(found)) if found else None
    summary["mean_rank_found"] = _round(np.mean(found)) if found else None

    # Misses are counted as (max_k + 1) for the global rank statistics.
    miss_value = max(ks) + 1 if ks else 1
    effective = [rank if rank is not None else miss_value for rank in ranks]
    summary["median_rank"] = _round(np.median(effective))
    summary["mean_rank"] = _round(np.mean(effective))
    return summary


def evaluate_text_to_image(
    image_embeddings: np.ndarray,
    image_meta: list[dict],
    caption_embeddings: np.ndarray,
    caption_meta: list[dict],
    *,
    ks: list[int],
    max_queries: int,
    seed: int = DEFAULT_SEED,
) -> dict:
    index = _flat_index(image_embeddings)
    max_k = min(max(ks), index.ntotal)
    query_ids = _sample_indices(len(caption_meta), max_queries, seed)
    if not query_ids:
        return {"n_queries": 0}

    queries = np.ascontiguousarray(caption_embeddings[query_ids], dtype=np.float32)
    _, ids = index.search(queries, max_k)

    image_ids = [int(record["image_id"]) for record in image_meta]
    ranks: list[int | None] = []
    for row, caption_id in enumerate(query_ids):
        ground_truth = int(caption_meta[caption_id]["image_id"])
        rank = None
        for position, vector_id in enumerate(ids[row], start=1):
            if vector_id < 0:
                continue
            if image_ids[int(vector_id)] == ground_truth:
                rank = position
                break
        ranks.append(rank)

    return _summarize_ranks(ranks, ks)


def evaluate_image_to_text(
    image_embeddings: np.ndarray,
    image_meta: list[dict],
    caption_embeddings: np.ndarray,
    caption_meta: list[dict],
    *,
    ks: list[int],
    max_queries: int,
    seed: int = DEFAULT_SEED,
) -> dict:
    index = _flat_index(caption_embeddings)
    max_k = min(max(ks), index.ntotal)
    query_ids = _sample_indices(len(image_meta), max_queries, seed)
    if not query_ids:
        return {"n_queries": 0}

    queries = np.ascontiguousarray(image_embeddings[query_ids], dtype=np.float32)
    _, ids = index.search(queries, max_k)

    caption_image_ids = [int(record["image_id"]) for record in caption_meta]
    ranks: list[int | None] = []
    for row, image_row in enumerate(query_ids):
        ground_truth = int(image_meta[image_row]["image_id"])
        rank = None
        for position, vector_id in enumerate(ids[row], start=1):
            if vector_id < 0:
                continue
            if caption_image_ids[int(vector_id)] == ground_truth:
                rank = position
                break
        ranks.append(rank)

    return _summarize_ranks(ranks, ks)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def measure_latency(
    *,
    model_name: str,
    image_index,
    queries: list[str],
    samples: int = 20,
    seed: int = DEFAULT_SEED,
) -> dict:
    if not queries:
        return {}

    rng = random.Random(seed)
    selected = queries if len(queries) <= samples else rng.sample(queries, samples)

    encoder = ClipEncoder(model_name)
    encoder.load()

    # Warm-up (model / FAISS caches).
    warmup = encoder.encode_texts([selected[0]], progress=False)
    image_index.search(np.ascontiguousarray(warmup, dtype=np.float32), 1)

    encode_times: list[float] = []
    search_times: list[float] = []
    for query in selected:
        started = time.perf_counter()
        vector = encoder.encode_texts([query], progress=False)
        encoded = time.perf_counter()
        image_index.search(np.ascontiguousarray(vector, dtype=np.float32), 5)
        finished = time.perf_counter()
        encode_times.append((encoded - started) * 1000)
        search_times.append((finished - encoded) * 1000)

    def stats(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": _round(np.mean(array), 2),
            "p50_ms": _round(np.percentile(array, 50), 2),
            "p95_ms": _round(np.percentile(array, 95), 2),
            "min_ms": _round(np.min(array), 2),
            "max_ms": _round(np.max(array), 2),
        }

    return {
        "samples": len(selected),
        "device": encoder.device,
        "text_encoding": stats(encode_times),
        "faiss_search": stats(search_times),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(str(item) for item in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _verdict(metrics: dict, min_recall1_heldout: float, min_recall1_kb: float) -> dict:
    failures: list[str] = []
    warnings: list[str] = []

    for key in ("images", "captions"):
        check = metrics["embeddings"][key]
        if not check["finite"]:
            failures.append(f"{key}: NaN/Inf values detected")
        if not check["metadata_matches"]:
            failures.append(f"{key}: embedding/metadata count mismatch")
        if check["zero_vectors"] > 0:
            failures.append(f"{key}: {check['zero_vectors']} zero vector(s)")
        if not check["unique_ids_ok"]:
            failures.append(f"{key}: duplicate IDs in metadata")
        if not check["vector_ids_sequential"]:
            failures.append(f"{key}: vector_id values are not sequential")
        if check["unit_norm_max_deviation"] is not None and check["unit_norm_max_deviation"] > 1e-3:
            warnings.append(
                f"{key}: L2 norm deviation {check['unit_norm_max_deviation']:.2e} > 1e-3"
            )
        if check["exact_duplicate_rows"] > 0:
            warnings.append(
                f"{key}: {check['exact_duplicate_rows']} exact duplicate vector(s)"
            )
        if check["near_duplicate_pairs"] > 0:
            warnings.append(
                f"{key}: {check['near_duplicate_pairs']} near-duplicate pair(s) "
                f"(cosine >= {check['near_duplicate_threshold']})"
            )

    metadata = metrics["metadata"]
    if metadata["images_without_captions"] > 0:
        failures.append(
            f"{metadata['images_without_captions']} image(s) without captions"
        )
    if metadata["captions_referencing_unknown_images"] > 0:
        failures.append(
            f"{metadata['captions_referencing_unknown_images']} caption(s) reference unknown images"
        )
    if metadata["missing_image_files"] > 0:
        warnings.append(f"{metadata['missing_image_files']} image file(s) missing")
    if metadata["lfs_pointer_files"] > 0:
        warnings.append(
            f"{metadata['lfs_pointer_files']} image file(s) are Git LFS pointers"
        )

    for key in ("images", "captions"):
        check = metrics["indices"][key]
        if not check["count_matches"]:
            failures.append(f"{key} index: FAISS count != metadata count")
        if not check["dimension_matches"]:
            failures.append(f"{key} index: wrong dimension")

    heldout = metrics.get("retrieval", {}).get("heldout")
    if heldout:
        recall1 = heldout["text_to_image"].get("recall@1")
        if recall1 is not None and recall1 < min_recall1_heldout:
            warnings.append(
                f"held-out text->image Recall@1={recall1:.3f} "
                f"< threshold {min_recall1_heldout:.3f}"
            )
    kb = metrics.get("retrieval", {}).get("knowledge_base")
    if kb:
        recall1 = kb["text_to_image"].get("recall@1")
        if recall1 is not None and recall1 < min_recall1_kb:
            warnings.append(
                f"KB text->image Recall@1={recall1:.3f} "
                f"< threshold {min_recall1_kb:.3f}"
            )

    overall = "FAIL" if failures else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "overall": overall,
        "failures": failures,
        "warnings": warnings,
    }


def build_report(metrics: dict) -> str:
    config = metrics["configuration"]
    lines: list[str] = []
    add = lines.append

    add("# Quality Control Report - Embeddings & Vector Database")
    add("")
    add(f"**Generated:** {metrics['generated_at']}  ")
    add("**Task:** Yahia - Embeddings, Vector Database & Quality Control  ")
    add(f"**Overall verdict:** **{metrics['verdict']['overall']}**")
    add("")
    add("## 1. Configuration")
    add("")
    add(_table(
        ["Parameter", "Value"],
        [
            ["Model", config["model"]],
            ["Embedding dimension", config["dimension"]],
            ["Similarity metric", config["metric"]],
            ["Knowledge Base split", config["kb_split"]],
            ["Knowledge Base images", config["kb_images"]],
            ["Knowledge Base captions", config["kb_captions"]],
            ["Evaluation split", config["eval_split"] or "not available"],
            ["Evaluation images", config["eval_images"]],
            ["Evaluation captions", config["eval_captions"]],
            ["Index type", config["index_type"]],
            ["Seed", config["seed"]],
        ],
    ))
    add("")

    if config.get("subset_run"):
        add(
            "> **Note:** this run used a subset of the Knowledge Base "
            f"({config['kb_images']} images). Re-run with `--kb-limit 0` for the "
            "full 10,000-image split. Subset metrics are not directly comparable "
            "with full-split metrics."
        )
        add("")

    add("## 2. Embedding integrity")
    add("")
    rows = []
    for key in ("images", "captions"):
        check = metrics["embeddings"][key]
        rows.append([
            key,
            check["count"],
            check["dimension"],
            check["finite"],
            check["zero_vectors"],
            check["norm_mean"],
            check["unit_norm_max_deviation"],
            check["exact_duplicate_rows"],
            check["near_duplicate_pairs"],
        ])
    add(_table(
        [
            "Vectors",
            "Count",
            "Dim",
            "Finite",
            "Zero",
            "Norm mean",
            "Max norm dev.",
            "Exact dups",
            "Near dups",
        ],
        rows,
    ))
    add("")
    add(
        f"Near-duplicate check: cosine threshold "
        f"{metrics['embeddings']['images']['near_duplicate_threshold']}, "
        f"sample size {metrics['embeddings']['images']['near_duplicate_sample_size']}."
    )
    add("")

    caption_stats = metrics["embeddings"]["captions"]["caption_stats"]
    if caption_stats:
        add("### Caption text statistics")
        add("")
        add(_table(
            ["Metric", "Value"],
            [
                ["Empty captions", caption_stats["empty_captions"]],
                ["Words min", caption_stats["words_min"]],
                ["Words mean", caption_stats["words_mean"]],
                ["Words median", caption_stats["words_median"]],
                ["Words max", caption_stats["words_max"]],
            ],
        ))
        add("")

    add("## 3. Metadata and coverage")
    add("")
    metadata = metrics["metadata"]
    add(_table(
        ["Check", "Value"],
        [
            ["Images", metadata["images"]],
            ["Captions", metadata["captions"]],
            ["Unique image IDs", metadata["unique_image_ids"]],
            ["Captions referencing unknown images", metadata["captions_referencing_unknown_images"]],
            ["Images without captions", metadata["images_without_captions"]],
            ["Average captions per image", metadata["average_captions_per_image"]],
            ["Duplicate caption texts", metadata["duplicate_caption_texts"]],
            ["Duplicate caption rate", metadata["duplicate_caption_rate"]],
            ["Missing image files", metadata["missing_image_files"]],
            ["Git LFS pointer files", metadata["lfs_pointer_files"]],
        ],
    ))
    add("")
    distribution = metadata["captions_per_image_distribution"]
    if distribution:
        add("Captions per image: " + ", ".join(
            f"{count} caption(s): {images} image(s)"
            for count, images in distribution.items()
        ))
        add("")

    add("## 4. Vector database integrity")
    add("")
    rows = []
    for key in ("images", "captions", "joint"):
        check = metrics["indices"][key]
        rows.append([
            key,
            check["index_type"],
            check["count"],
            check["metadata_count"],
            check["dimension"],
            check["self_retrieval_accuracy"],
            f"{check['file_size_bytes'] / 1024:.1f} KB",
        ])
    add(_table(
        [
            "Index",
            "Type",
            "Vectors",
            "Metadata rows",
            "Dim",
            "Self top-1",
            "Size",
        ],
        rows,
    ))
    add("")
    add(
        "Self top-1 is the fraction of sampled vectors whose nearest neighbour "
        "is the vector itself (sanity check of the index)."
    )
    add("")

    add("## 5. Retrieval quality")
    add("")
    retrieval = metrics.get("retrieval", {})
    for label, key in (
        ("Held-out evaluation split", "heldout"),
        ("Knowledge Base self-retrieval", "knowledge_base"),
    ):
        block = retrieval.get(key)
        if not block:
            add(f"### {label}")
            add("")
            add("_Not evaluated: embeddings for this split are not available._")
            add("")
            continue

        add(f"### {label}")
        add("")
        rows = []
        for direction, name in (
            ("text_to_image", "Text -> image"),
            ("image_to_text", "Image -> text"),
        ):
            result = block[direction]
            rows.append([
                name,
                result.get("n_queries"),
                result.get("recall@1"),
                result.get("recall@5"),
                result.get("recall@10"),
                result.get("mrr"),
                result.get("median_rank"),
            ])
        add(_table(
            ["Direction", "Queries", "R@1", "R@5", "R@10", "MRR", "Median rank"],
            rows,
        ))
        add("")

    add("## 6. Performance")
    add("")
    latency = metrics.get("latency") or {}
    if latency:
        add(_table(
            ["Operation", "Mean (ms)", "p50 (ms)", "p95 (ms)", "Min (ms)", "Max (ms)"],
            [
                [
                    "CLIP text encoding",
                    latency["text_encoding"]["mean_ms"],
                    latency["text_encoding"]["p50_ms"],
                    latency["text_encoding"]["p95_ms"],
                    latency["text_encoding"]["min_ms"],
                    latency["text_encoding"]["max_ms"],
                ],
                [
                    "FAISS search (top-5)",
                    latency["faiss_search"]["mean_ms"],
                    latency["faiss_search"]["p50_ms"],
                    latency["faiss_search"]["p95_ms"],
                    latency["faiss_search"]["min_ms"],
                    latency["faiss_search"]["max_ms"],
                ],
            ],
        ))
        add("")
        add(f"Measured on {latency['samples']} queries ({latency['device']}).")
        add("")
    else:
        add("_Latency measurement was skipped._")
        add("")

    add("## 7. Verdict")
    add("")
    verdict = metrics["verdict"]
    add(f"**Overall: {verdict['overall']}**")
    add("")
    if verdict["failures"]:
        add("**Blocking issues**")
        add("")
        for failure in verdict["failures"]:
            add(f"- {failure}")
        add("")
    if verdict["warnings"]:
        add("**Warnings**")
        add("")
        for warning in verdict["warnings"]:
            add(f"- {warning}")
        add("")
    if metrics["embeddings"]["captions"]["exact_duplicate_rows"] > 0:
        add(
            "> **Note:** duplicate caption vectors come from duplicate caption "
            "texts in COCO (the same sentence is sometimes reused for an image). "
            "They are reported for transparency and do not block the pipeline."
        )
        add("")
    if not verdict["failures"] and not verdict["warnings"]:
        add("All automated checks passed.")
        add("")

    add("## 8. Limitations and recommendations")
    add("")
    add(
        "- CLIP ViT-B/32 is used **zero-shot** (no fine-tuning on COCO). "
        "Retrieval scores are therefore a baseline, not a ceiling."
    )
    add(
        "- Recall@K on a small gallery is optimistic: with a 100-image gallery, "
        "random chance is already 1%. Compare runs with the same gallery size."
    )
    add(
        "- The repository originally stores processed images as Git LFS pointers. "
        "The pipeline downloads the real COCO images from `coco_url` when "
        "`--download` is used; this requires an internet connection."
    )
    add(
        "- For production, consider fine-tuning CLIP on COCO captions, using a "
        "larger encoder (ViT-L/14) and an approximate index (IVF/HNSW) for "
        "hundreds of thousands of vectors."
    )
    add(
        "- Re-run the full Knowledge Base (`--kb-limit 0`) before reporting final "
        "numbers to the team."
    )
    add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_quality_control(
    *,
    kb_split: str = "knowledge_base",
    eval_split: str | None = "test",
    ks: list[int] | None = None,
    max_queries: int = 200,
    near_duplicate_threshold: float = 0.999,
    model_name: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    min_recall1_heldout: float = 0.30,
    min_recall1_kb: float = 0.10,
    measure_latency_flag: bool = True,
    report_path: Path | None = None,
    metrics_path: Path | None = None,
) -> dict:
    ensure_dirs()
    ks = sorted(set(ks or [1, 5, 10]))
    report_path = report_path or (REPORTS_DIR / "qc_report.md")
    metrics_path = metrics_path or (REPORTS_DIR / "qc_metrics.json")

    print("\n=== Quality control ===")
    kb = load_embedding_artifacts(kb_split)
    db_manifest_path = VECTOR_DB_DIR / "manifest.json"
    if not db_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {db_manifest_path}. Run build_vector_db.py first."
        )
    db_manifest = read_json(db_manifest_path)

    if db_manifest["source_split"] != kb_split:
        raise ValueError(
            f"The vector database was built from '{db_manifest['source_split']}', "
            f"but quality control requested '{kb_split}'."
        )

    image_checks = check_embeddings(
        "images",
        kb["image_embeddings"],
        kb["image_meta"],
        id_key="image_id",
        near_duplicate_threshold=near_duplicate_threshold,
        seed=seed,
    )
    caption_checks = check_embeddings(
        "captions",
        kb["caption_embeddings"],
        kb["caption_meta"],
        id_key="caption_id",
        near_duplicate_threshold=near_duplicate_threshold,
        seed=seed,
    )
    metadata_checks = check_metadata(
        kb_split, kb["image_meta"], kb["caption_meta"]
    )

    index_checks = {}
    for key in ("images", "captions", "joint"):
        entry = db_manifest["indices"][key]
        index_path = resolve_project_path(entry["path"])
        if key == "joint":
            metadata = read_jsonl(resolve_project_path(entry["metadata"]))
            embeddings = np.vstack(
                [kb["image_embeddings"], kb["caption_embeddings"]]
            )
        else:
            metadata = kb["image_meta"] if key == "images" else kb["caption_meta"]
            embeddings = (
                kb["image_embeddings"] if key == "images" else kb["caption_embeddings"]
            )
        index_checks[key] = check_index(
            key,
            index_path,
            metadata,
            embeddings,
            expected_dimension=int(db_manifest["embedding_dim"]),
            seed=seed,
        )

    retrieval: dict[str, dict] = {
        "knowledge_base": {
            "text_to_image": evaluate_text_to_image(
                kb["image_embeddings"],
                kb["image_meta"],
                kb["caption_embeddings"],
                kb["caption_meta"],
                ks=ks,
                max_queries=max_queries,
                seed=seed,
            ),
            "image_to_text": evaluate_image_to_text(
                kb["image_embeddings"],
                kb["image_meta"],
                kb["caption_embeddings"],
                kb["caption_meta"],
                ks=ks,
                max_queries=max_queries,
                seed=seed,
            ),
        }
    }

    eval_images = 0
    eval_captions = 0
    if eval_split:
        try:
            evaluation = load_embedding_artifacts(eval_split)
        except FileNotFoundError:
            print(
                f"[qc] no embeddings for evaluation split '{eval_split}', "
                "skipping held-out retrieval."
            )
        else:
            eval_images = len(evaluation["image_meta"])
            eval_captions = len(evaluation["caption_meta"])
            retrieval["heldout"] = {
                "split": eval_split,
                "text_to_image": evaluate_text_to_image(
                    evaluation["image_embeddings"],
                    evaluation["image_meta"],
                    evaluation["caption_embeddings"],
                    evaluation["caption_meta"],
                    ks=ks,
                    max_queries=max_queries,
                    seed=seed,
                ),
                "image_to_text": evaluate_image_to_text(
                    evaluation["image_embeddings"],
                    evaluation["image_meta"],
                    evaluation["caption_embeddings"],
                    evaluation["caption_meta"],
                    ks=ks,
                    max_queries=max_queries,
                    seed=seed,
                ),
            }

    latency = {}
    if measure_latency_flag:
        image_index = faiss.read_index(
            str(resolve_project_path(db_manifest["indices"]["images"]["path"]))
        )
        sample_queries = [
            record["caption"]
            for record in kb["caption_meta"][
                : min(200, len(kb["caption_meta"]))
            ]
        ]
        latency = measure_latency(
            model_name=model_name,
            image_index=image_index,
            queries=sample_queries,
            samples=20,
            seed=seed,
        )

    configuration = {
        "model": db_manifest.get("model", model_name),
        "dimension": int(db_manifest["embedding_dim"]),
        "metric": db_manifest["metric"],
        "index_type": db_manifest["index_type"],
        "kb_split": kb_split,
        "kb_images": image_checks["count"],
        "kb_captions": caption_checks["count"],
        "eval_split": eval_split,
        "eval_images": eval_images,
        "eval_captions": eval_captions,
        "seed": seed,
        "subset_run": bool(kb["manifest"].get("limit")),
    }

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "embeddings": {
            "images": image_checks,
            "captions": caption_checks,
        },
        "metadata": metadata_checks,
        "indices": index_checks,
        "retrieval": retrieval,
        "latency": latency,
    }
    metrics["verdict"] = _verdict(
        metrics, min_recall1_heldout, min_recall1_kb
    )

    from .data import write_json

    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, metrics)
    report_path.write_text(build_report(metrics), encoding="utf-8")

    print(f"QC verdict: {metrics['verdict']['overall']}")
    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality control on embeddings + vector DB.")
    parser.add_argument("--kb-split", default="knowledge_base")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--k", default="1,5,10")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.999)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-recall1-heldout", type=float, default=0.30)
    parser.add_argument("--min-recall1-kb", type=float, default=0.10)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--metrics", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = [int(item) for item in args.k.split(",") if item.strip()]
    run_quality_control(
        kb_split=args.kb_split,
        eval_split=args.eval_split or None,
        ks=ks,
        max_queries=args.max_queries,
        near_duplicate_threshold=args.near_duplicate_threshold,
        model_name=args.model,
        seed=args.seed,
        min_recall1_heldout=args.min_recall1_heldout,
        min_recall1_kb=args.min_recall1_kb,
        measure_latency_flag=not args.skip_latency,
        report_path=Path(args.report) if args.report else None,
        metrics_path=Path(args.metrics) if args.metrics else None,
    )


if __name__ == "__main__":
    main()
