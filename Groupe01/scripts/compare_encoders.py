"""Compare frozen encoders using chunked NumPy exact inner-product search.

Reported latency is for NumPy exact search, not FAISS, and is not comparable
to the latency measurements in artifacts/index/index_report.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocoevalcap.cider.cider import Cider

from common import clean_caption

ARTIFACTS = ROOT / "artifacts"
CONFIGS = [
    ("clip-b32", "img2img"),
    ("clip-b32", "img2txt"),
    ("dinov2-s", "img2img"),
    ("siglip-b16", "img2img"),
    ("siglip-b16", "img2txt"),
]
EXPECTED_VALIDATION_CIDER = {
    "siglip-b16|img2txt": 0.7222567866866058,
    "siglip-b16|img2img": 0.6735370593982103,
    "clip-b32|img2txt": 0.5584101764891685,
    "dinov2-s|img2img": 0.5541039139825013,
    "clip-b32|img2img": 0.5285707244987701,
}
CSV_FIELDS = [
    "dataset",
    "report_note",
    "config",
    "encoder",
    "mode",
    "n_queries",
    "queries_without_objects",
    "relevance_queries",
    "recall@1",
    "recall@5",
    "recall@10",
    "precision@1",
    "precision@5",
    "precision@10",
    "mrr",
    "cider",
    "random_caption_cider_seed42",
    "latency_ms_per_query_median_5_repeats",
]
CSV_LATENCY_COMMENT = (
    "# Latency is median of 5 repeats of chunked NumPy exact inner-product search, "
    "not FAISS; it is not comparable to artifacts/index/index_report.json.\n"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_metadata() -> dict[int, dict]:
    return {int(row["image_id"]): row for row in read_jsonl(ARTIFACTS / "tables" / "metadata.jsonl")}


def classes_for(metadata: dict[int, dict], image_id: int) -> set[str]:
    row = metadata.get(int(image_id), {})
    return {
        str(category)
        for category, count in row.get("category_counts", {}).items()
        if int(count) > 0
    }


def cider_per_image(predictions: dict[int, str], references: dict[int, list[str]]) -> np.ndarray:
    ids = sorted(references)
    _, scores = Cider().compute_score(
        {image_id: [clean_caption(text) for text in references[image_id]] for image_id in ids},
        {image_id: [clean_caption(predictions[image_id])] for image_id in ids},
    )
    return np.asarray(scores, dtype=np.float64)


def exact_chunked_search(
    query_vectors: np.ndarray,
    database: np.ndarray,
    top_k: int,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return exact top-k inner products, breaking ties by lowest database index."""
    if query_vectors.ndim != 2 or database.ndim != 2:
        raise ValueError("Query and database embeddings must be rank-2 arrays")
    if query_vectors.shape[1] != database.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ: {query_vectors.shape[1]} vs {database.shape[1]}"
        )
    if not 0 < top_k <= len(database):
        raise ValueError(f"top_k must be between 1 and database size ({len(database)})")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    def search_once() -> tuple[np.ndarray, np.ndarray]:
        best_scores = np.empty((len(query_vectors), 0), dtype=np.float32)
        best_indices = np.empty((len(query_vectors), 0), dtype=np.int64)
        for start in range(0, len(database), chunk_size):
            stop = min(start + chunk_size, len(database))
            # Chunking bounds the temporary similarity matrix and avoids FAISS's
            # large text-index allocation failure on this Windows environment.
            chunk = np.asarray(database[start:stop], dtype=np.float32)
            chunk_scores = query_vectors @ chunk.T
            chunk_indices = np.broadcast_to(
                np.arange(start, stop, dtype=np.int64),
                (len(query_vectors), stop - start),
            )
            candidate_scores = np.concatenate((best_scores, chunk_scores), axis=1)
            candidate_indices = np.concatenate((best_indices, chunk_indices), axis=1)
            order = np.lexsort((candidate_indices, -candidate_scores), axis=1)[:, :top_k]
            best_scores = np.take_along_axis(candidate_scores, order, axis=1)
            best_indices = np.take_along_axis(candidate_indices, order, axis=1)
        return best_scores, best_indices

    search_once()
    durations = []
    first_result = None
    for repeat in range(5):
        start = time.perf_counter()
        result = search_once()
        durations.append(time.perf_counter() - start)
        if repeat == 0:
            first_result = result
    assert first_result is not None
    return (
        first_result[0],
        first_result[1],
        float(np.median(durations) * 1000 / len(query_vectors)),
    )


def retrieve(
    encoder: str,
    mode: str,
    query_vectors: np.ndarray,
    top_k: int,
    kb_captions: list[dict],
    metadata: dict[int, dict],
):
    emb_dir = ARTIFACTS / "embeddings" / encoder
    kb_ids = np.load(emb_dir / "ids_kb.npy")
    kb_by_image: dict[int, list[dict]] = defaultdict(list)
    captions_by_id = {}
    for row in kb_captions:
        image_id = int(row["image_id"])
        caption_id = int(row["caption_id"])
        kb_by_image[image_id].append(row)
        captions_by_id[caption_id] = row["caption_raw"]

    if mode == "img2img":
        database = np.load(emb_dir / "kb_img.npy", mmap_mode="r")
        if len(database) != len(kb_ids):
            raise ValueError(f"{encoder} image vectors and KB IDs have different lengths")
        scores, offsets, latency_ms = exact_chunked_search(query_vectors, database, top_k)
        retrieved_images = kb_ids[offsets]
        retrieved_classes = [
            [classes_for(metadata, int(image_id)) for image_id in row]
            for row in retrieved_images
        ]
        predictions = {}
        for row_idx, image_ids in enumerate(retrieved_images):
            for image_id in image_ids[:5]:
                options = kb_by_image.get(int(image_id), [])
                if options:
                    predictions[row_idx] = options[0]["caption_raw"]
                    break
            else:
                raise RuntimeError(f"{encoder} img2img returned no captioned KB image for query row {row_idx}")
        return retrieved_classes, predictions, latency_ms
    if mode == "img2txt":
        database = np.load(emb_dir / "kb_txt.npy", mmap_mode="r")
        caption_ids = np.load(emb_dir / "kb_txt_caption_ids.npy")
        parent_ids = np.load(emb_dir / "kb_txt_image_ids.npy")
        if len(database) != len(caption_ids) or len(caption_ids) != len(parent_ids):
            raise ValueError(f"{encoder} caption vectors, caption IDs, and parent IDs differ in length")
        scores, offsets, latency_ms = exact_chunked_search(query_vectors, database, top_k)
        retrieved_caption_ids = caption_ids[offsets]
        retrieved_parent_ids = parent_ids[offsets]
        retrieved_classes = [
            [classes_for(metadata, int(image_id)) for image_id in row]
            for row in retrieved_parent_ids
        ]
        predictions = {
            row_idx: captions_by_id[int(caption_id)]
            for row_idx, caption_id in enumerate(retrieved_caption_ids[:, 0])
        }
        return retrieved_classes, predictions, latency_ms

    raise ValueError(f"Unsupported retrieval mode: {mode}")


def relevance_metrics(
    query_ids: np.ndarray,
    retrieved_classes: list[list[set[str]]],
    metadata: dict[int, dict],
) -> tuple[int, dict[str, float]]:
    eligible = [
        row_idx
        for row_idx, image_id in enumerate(query_ids)
        if classes_for(metadata, int(image_id))
    ]
    result = {}
    for k in (1, 5, 10):
        hits = []
        precisions = []
        reciprocal_ranks = []
        for row_idx in eligible:
            query_classes = classes_for(metadata, int(query_ids[row_idx]))
            relevant = [
                bool(query_classes & retrieved_classes[row_idx][rank])
                for rank in range(k)
            ]
            hits.append(any(relevant))
            precisions.append(sum(relevant) / k)
            first = next((rank + 1 for rank, is_relevant in enumerate(relevant) if is_relevant), None)
            reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
        result[f"recall@{k}"] = float(np.mean(hits)) if eligible else 0.0
        result[f"precision@{k}"] = float(np.mean(precisions)) if eligible else 0.0
        if k == 10:
            result["mrr"] = float(np.mean(reciprocal_ranks)) if eligible else 0.0
    return len(query_ids) - len(eligible), result


def evaluate_split(split: str, metadata: dict[int, dict], kb_captions: list[dict]) -> list[dict]:
    rows = []
    references_rows = read_jsonl(ARTIFACTS / "tables" / f"eval_refs_{split}.jsonl")
    references: dict[int, list[str]] = defaultdict(list)
    for row in references_rows:
        references[int(row["image_id"])].append(row["caption_raw"])
    if len(references) != 100:
        raise ValueError(f"Expected 100 {split} query images, found {len(references)}")

    # One reproducible random KB caption per query is shared across configurations.
    rng = np.random.RandomState(42)
    random_predictions = {
        image_id: kb_captions[int(rng.randint(0, len(kb_captions)))]["caption_raw"]
        for image_id in sorted(references)
    }
    random_cider = float(np.mean(cider_per_image(random_predictions, references)))
    print(f"Evaluating {split}: {len(references)} queries; random baseline CIDEr={random_cider:.9f}", flush=True)
    query_ids_by_encoder = {}
    for encoder, _ in CONFIGS:
        if encoder not in query_ids_by_encoder:
            emb_dir = ARTIFACTS / "embeddings" / encoder
            query_ids = np.load(emb_dir / f"ids_{split}.npy")
            query_vectors = np.load(emb_dir / f"{split}_img.npy")
            if len(query_ids) != len(query_vectors) or len(query_ids) != len(references):
                raise ValueError(f"{encoder} {split} IDs and vectors are not aligned with references")
            query_ids_by_encoder[encoder] = (query_ids, query_vectors)

    for encoder, mode in CONFIGS:
        print(f"  {split}: {encoder}|{mode}", flush=True)
        query_ids, query_vectors = query_ids_by_encoder[encoder]
        retrieved_classes, predictions_by_row, latency_ms = retrieve(
            encoder,
            mode,
            np.ascontiguousarray(query_vectors, dtype=np.float32),
            10,
            kb_captions,
            metadata,
        )
        predictions = {
            int(query_ids[row_idx]): caption
            for row_idx, caption in predictions_by_row.items()
        }
        if set(predictions) != set(references):
            raise ValueError(f"{encoder} {mode} returned incomplete predictions for {split}")
        no_objects, metrics = relevance_metrics(query_ids, retrieved_classes, metadata)
        cider = float(np.mean(cider_per_image(predictions, references)))
        rows.append(
            {
                "dataset": "validation" if split == "val" else "test",
                "report_note": "" if split == "val" else "report only, not used for selection",
                "config": f"{encoder}|{mode}",
                "encoder": encoder,
                "mode": mode,
                "n_queries": len(query_ids),
                "queries_without_objects": no_objects,
                "relevance_queries": len(query_ids) - no_objects,
                **metrics,
                "cider": cider,
                "random_caption_cider_seed42": random_cider,
                "latency_ms_per_query_median_5_repeats": latency_ms,
            }
        )
        print(
            f"    CIDEr={cider:.9f}; no-object queries={no_objects}; "
            f"median latency={latency_ms:.4f} ms/query",
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(CSV_LATENCY_COMMENT)
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_figure(rows: list[dict], path: Path) -> None:
    ordered = [f"{encoder}|{mode}" for encoder, mode in CONFIGS]
    by_config = {row["config"]: row for row in rows}
    labels = [config.replace("|", "\n") for config in ordered]
    x = np.arange(len(ordered))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cider_bars = ax.bar(x - width / 2, [by_config[c]["cider"] for c in ordered], width, label="CIDEr")
    mrr_bars = ax.bar(x + width / 2, [by_config[c]["mrr"] for c in ordered], width, label="MRR")
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.bar_label(cider_bars, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(mrr_bars, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(sys.executable, flush=True)
    metadata = load_metadata()
    kb_captions = read_jsonl(ARTIFACTS / "tables" / "kb_captions.jsonl")
    if len(kb_captions) != 50000:
        raise ValueError(f"Expected 50000 KB captions, found {len(kb_captions)}")

    val_rows = evaluate_split("val", metadata, kb_captions)
    mismatches = [
        (row["config"], row["cider"], EXPECTED_VALIDATION_CIDER[row["config"]])
        for row in val_rows
        if abs(row["cider"] - EXPECTED_VALIDATION_CIDER[row["config"]]) > 1e-6
    ]
    if mismatches:
        raise RuntimeError(f"Validation CIDEr does not match selection report: {mismatches}")

    test_rows = evaluate_split("test", metadata, kb_captions)
    write_csv(ROOT / "reports" / "tables" / "encoder_comparison_val.csv", val_rows)
    write_csv(ROOT / "reports" / "tables" / "encoder_comparison_test.csv", test_rows)
    write_figure(val_rows, ROOT / "reports" / "figures" / "encoder_comparison_val.png")

    print(
        "Relevance: a retrieved KB image is relevant when it shares at least one "
        "object class with the query image in metadata.jsonl; for img2txt, the "
        "retrieved caption's parent image is used. Queries with no metadata "
        "objects are excluded from relevance metrics only."
    )
    print(CSV_LATENCY_COMMENT.strip("# \n"))
    for dataset, rows in (("validation", val_rows), ("test (report only, not used for selection)", test_rows)):
        print(f"\n{dataset}")
        print(",".join(CSV_FIELDS))
        for row in rows:
            print(",".join(str(row[field]) for field in CSV_FIELDS))
    print("\nCreated reports/tables/encoder_comparison_val.csv")
    print("Created reports/tables/encoder_comparison_test.csv")
    print("Created reports/figures/encoder_comparison_val.png")


if __name__ == "__main__":
    main()
