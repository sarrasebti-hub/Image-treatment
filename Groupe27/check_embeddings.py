"""Quality control, leakage audit and retrieval sanity check for the embeddings/ release.

Hard checks must pass, otherwise the exit code is 1: array shapes, finite unit vectors,
row-to-ID maps, caption table, checksums, FAISS contents, split leakage, self-exclusion,
FAISS = brute force and re-encoding consistency. Informational results: exact-text
overlap between test and KB captions, a near-duplicate scan, validation retrieval
metrics and latency.

Retrieval quality is measured on the 100 validation queries only. The test split is used
only for the leakage audit: its captions are read to prove they are absent, never embedded.

Writes embeddings/qc_report.json and embeddings/figures/*.png.
Usage: python check_embeddings.py [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from retrieval import (  # noqa: E402
    DATA_DIR,
    EMBED_DIR,
    KnowledgeBase,
    load_rgb_image,
    normalize_caption,
    read_json,
    sha256_file,
    write_json,
)

logger = logging.getLogger("check_embeddings")
NEAR_DUPLICATE_COSINE = 0.95   # pairs at or above this CLIP image cosine are flagged for manual review
UNIT_NORM_TOLERANCE = 1e-4
REENCODE_MIN_COSINE = 0.9999
SEED = 42
INK, MUTED = "#1f1f1f", "#5f5f5f"


class Checks:
    """Named pass/fail checks with their evidence."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, **evidence: Any) -> None:
        """Record one check and log it."""
        self.results.append({"check": name, "passed": bool(passed), **evidence})
        (logger.info if passed else logger.error)("%s: %s", "PASS" if passed else "FAIL", name)

    @property
    def passed(self) -> bool:
        """True when every recorded check passed."""
        return all(result["passed"] for result in self.results)


def check_structure(kb: KnowledgeBase, checks: Checks, data_dir: Path, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """Shapes, finite unit vectors, ID maps, caption table, checksums and FAISS contents."""
    arrays = {"kb_images": kb.image_embeddings("knowledge_base"), "test_images": kb.image_embeddings("test"),
              "kb_captions_clip": kb.caption_embeddings("clip"), "kb_captions_sbert": kb.caption_embeddings("sbert")}
    kb_ids, val_ids, test_ids = (read_json(data_dir / name / "image_ids.json")
                                 for name in ("knowledge_base", "validation", "test"))
    clip_dim, sbert_dim = (meta["encoders"][name]["embedding_dim"] for name in ("clip", "sbert"))
    n_captions = len(kb.caption_ids)
    expected = {"kb_images": (len(kb_ids), clip_dim), "test_images": (len(test_ids), clip_dim),
                "kb_captions_clip": (n_captions, clip_dim), "kb_captions_sbert": (n_captions, sbert_dim)}
    for name, matrix in arrays.items():
        error = float(np.abs(np.linalg.norm(matrix, axis=1) - 1).max())
        checks.add(f"{name}: shape {expected[name]}, float32, no NaN/Inf, unit norm",
                   matrix.shape == expected[name] and matrix.dtype == np.float32
                   and bool(np.isfinite(matrix).all()) and error < UNIT_NORM_TOLERANCE,
                   shape=list(matrix.shape), max_norm_error=error)
    checks.add("kb_image_ids.json equals the KB split IDs (same order, no duplicates)",
               kb.kb_image_ids.tolist() == kb_ids and len(set(kb_ids)) == len(kb_ids))
    checks.add("test_image_ids.json equals the test split IDs", kb.test_image_ids.tolist() == test_ids)
    checks.add("validation_kb_rows.json points at the validation images inside the KB",
               sorted(kb.validation_image_ids.tolist()) == sorted(val_ids)
               and kb.kb_image_ids[kb.validation_kb_rows].tolist() == kb.validation_image_ids.tolist())

    by_id = {a["id"]: a for a in read_json(data_dir / "knowledge_base" / "captions.json")["annotations"]}
    with (kb.embed_dir / "kb_captions.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    table_ok = (len(table) == len(by_id) == len(set(kb.caption_ids.tolist()))
                and [int(r["row"]) for r in table] == list(range(len(table)))
                and all(int(r["caption_id"]) in by_id
                        and by_id[int(r["caption_id"])]["image_id"] == int(r["image_id"])
                        and by_id[int(r["caption_id"])]["caption"] == r["caption_raw"]
                        and r["caption_embedded"] == normalize_caption(r["caption_raw"]) for r in table))
    checks.add("kb_captions.csv: one row per KB caption; IDs and raw text match captions.json", table_ok,
               rows=len(table), kb_images_with_captions=len(set(kb.caption_image_ids.tolist())))

    files = meta["artifacts"] + meta["indexes"]
    mismatched = [f["path"] for f in files if sha256_file(kb.embed_dir / f["path"]) != f["sha256"]]
    checks.add("binary files match the SHA256 recorded in metadata.json", not mismatched,
               files_checked=len(files), mismatched=mismatched)
    inputs = meta["dataset"]["input_sha256"]
    changed = [name for name, digest in inputs.items() if sha256_file(data_dir / name.split("/", 1)[1]) != digest]
    checks.add("cleaned_dataset split files are unchanged since the build (SHA256)", not changed,
               files_checked=len(inputs), changed=changed)
    pairs = [(kb.image_index, arrays["kb_images"]), (kb.caption_indexes["clip"], arrays["kb_captions_clip"]),
             (kb.caption_indexes["sbert"], arrays["kb_captions_sbert"])]
    checks.add("FAISS indexes hold exactly the .npy vectors with the inner-product metric",
               all(index.ntotal == len(m) and index.d == m.shape[1] and index.metric_type == faiss.METRIC_INNER_PRODUCT
                   and np.array_equal(index.reconstruct_n(0, index.ntotal), m) for index, m in pairs))
    return arrays


def nearest_pairs(sims: np.ndarray, query_ids: np.ndarray, kb_ids: np.ndarray) -> dict[str, Any]:
    """Each query's most similar KB image: summary, flagged pairs and the top 10 pairs."""
    best_rows = sims.argmax(1)
    best = sims[np.arange(len(sims)), best_rows]
    pairs = [{"query_image_id": int(query_ids[i]), "kb_image_id": int(kb_ids[best_rows[i]]),
              "cosine": round(float(best[i]), 4)} for i in np.argsort(-best)]
    return {"max_cosine": pairs[0]["cosine"], "median_nearest_cosine": round(float(np.median(best)), 4),
            "flagged": [p for p in pairs if p["cosine"] >= NEAR_DUPLICATE_COSINE], "top_pairs": pairs[:10]}


def first_caption_rows(kb: KnowledgeBase) -> np.ndarray:
    """Caption-table row of the lowest caption ID of each validation image."""
    return np.array([np.flatnonzero(kb.caption_image_ids == v)[0] for v in kb.validation_image_ids])


def leakage_audit(kb: KnowledgeBase, checks: Checks, data_dir: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Split disjointness, caption leakage, self-exclusion, exact-text overlap and near-duplicates."""
    kb_ids, test_ids = set(kb.kb_image_ids.tolist()), set(kb.test_image_ids.tolist())
    val_ids = [int(v) for v in kb.validation_image_ids]
    test_captions = read_json(data_dir / "test" / "captions.json")["annotations"]
    checks.add("test images are not in the knowledge base", not kb_ids & test_ids)
    checks.add("validation images are all inside the knowledge base", set(val_ids) <= kb_ids)
    checks.add("no test caption ID and no test image ID appears in the KB caption table",
               not {a["id"] for a in test_captions} & set(kb.caption_ids.tolist())
               and not test_ids & set(kb.caption_image_ids.tolist()))

    queries, own = arrays["kb_images"][kb.validation_kb_rows], [{v} for v in val_ids]
    results = (kb.search_images(queries, k=10, exclude=own),
               kb.search_captions(queries, k=10, space="clip", exclude=own),
               kb.search_captions(arrays["kb_captions_sbert"][first_caption_rows(kb)], k=10, space="sbert",
                                  exclude=own))
    checks.add("self-exclusion: validation queries never get their own image or captions back (k=10)",
               all(len(hits) == 10 and all(h.image_id != v for h in hits)
                   for found in results for hits, v in zip(found, val_ids, strict=True)))

    kb_text: dict[str, set[int]] = {}
    for image_id, text in zip(kb.caption_image_ids.tolist(), kb.caption_texts, strict=True):
        kb_text.setdefault(normalize_caption(text).lower(), set()).add(image_id)
    overlap = [{"test_image_id": a["image_id"], "caption": a["caption"], "kb_image_ids": sorted(kb_text[key])}
               for a in test_captions if (key := normalize_caption(a["caption"]).lower()) in kb_text]
    val_sims = queries @ arrays["kb_images"].T
    val_sims[np.arange(len(val_ids)), kb.validation_kb_rows] = -np.inf
    return {
        "test_captions_with_identical_text_in_kb": {
            "count": len(overlap), "of": len(test_captions), "matches": overlap,
            "note": "Same wording written for different images (generic phrasings); not the test annotations. "
                    "KB captions are left unchanged: filtering the KB by test references would itself use them."},
        "near_duplicate_scan": {
            "method": "CLIP image cosine, every query against all 10,000 KB images",
            "review_threshold": NEAR_DUPLICATE_COSINE,
            "test_vs_kb": nearest_pairs(arrays["test_images"] @ arrays["kb_images"].T, kb.test_image_ids,
                                        kb.kb_image_ids),
            "validation_vs_other_kb": nearest_pairs(val_sims, kb.validation_image_ids, kb.kb_image_ids)},
    }


def reencode_consistency(kb: KnowledgeBase, checks: Checks, data_dir: Path,
                         arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Query-time encoding must reproduce the stored vectors (same code path as retrieval users)."""
    paths = [data_dir / "validation" / "images" / f"{i:012d}.jpg" for i in kb.validation_image_ids]
    image_cos = np.sum(kb.encode_images(paths) * arrays["kb_images"][kb.validation_kb_rows], axis=1)
    test_paths = [data_dir / "test" / "images" / f"{i:012d}.jpg" for i in kb.test_image_ids]
    test_cos = np.sum(kb.encode_images(test_paths) * arrays["test_images"], axis=1)
    rows = np.random.default_rng(SEED).choice(len(kb.caption_texts), 200, replace=False)
    texts = [kb.caption_texts[r] for r in rows]
    clip_cos = np.sum(kb.encode_texts(texts, "clip") * arrays["kb_captions_clip"][rows], axis=1)
    sbert_cos = np.sum(kb.encode_texts(texts, "sbert") * arrays["kb_captions_sbert"][rows], axis=1)
    result = {"validation_image_copies_min_cosine": float(image_cos.min()),
              "test_images_min_cosine": float(test_cos.min()),
              "clip_caption_min_cosine": float(clip_cos.min()), "sbert_caption_min_cosine": float(sbert_cos.min())}
    checks.add("re-encoding the 100 validation image copies reproduces their KB rows",
               result["validation_image_copies_min_cosine"] >= REENCODE_MIN_COSINE)
    checks.add("re-encoding the 100 test images reproduces their rows (test row-to-ID alignment)",
               result["test_images_min_cosine"] >= REENCODE_MIN_COSINE)
    checks.add("re-encoding 200 random KB captions reproduces the stored CLIP and SBERT rows",
               min(result["clip_caption_min_cosine"], result["sbert_caption_min_cosine"]) >= REENCODE_MIN_COSINE)
    return result


def rank_of(sims: np.ndarray, relevant: np.ndarray) -> int:
    """1-based rank of the best-scoring relevant item in a full ranking."""
    return int((sims > sims[relevant].max()).sum()) + 1


def recall(ranks: Sequence[int]) -> dict[str, float]:
    """Recall@1/5/10 (any relevant item in the top k), MRR and median rank."""
    r = np.asarray(ranks)
    return {"queries": int(len(r)), **{f"recall@{k}": round(float(np.mean(r <= k)), 4) for k in (1, 5, 10)},
            "mrr": round(float(np.mean(1 / r)), 4), "median_rank": float(np.median(r))}


def category_matrix(data_dir: Path, kb_ids: list[int]) -> np.ndarray:
    """Binary KB-image x COCO-category matrix from the KB instance annotations."""
    doc = read_json(data_dir / "knowledge_base" / "instances.json")
    column = {c["id"]: i for i, c in enumerate(sorted(doc["categories"], key=lambda c: c["id"]))}
    row = {image_id: r for r, image_id in enumerate(kb_ids)}
    matrix = np.zeros((len(kb_ids), len(column)), np.float32)
    for a in doc["annotations"]:
        matrix[row[a["image_id"]], column[a["category_id"]]] = 1
    return matrix


def validation_retrieval(kb: KnowledgeBase, checks: Checks, arrays: dict[str, np.ndarray],
                         categories: np.ndarray) -> dict[str, Any]:
    """Retrieval sanity metrics on the validation queries (relevance defined per task below)."""
    kb_img, clip_cap, sbert_cap = arrays["kb_images"], arrays["kb_captions_clip"], arrays["kb_captions_sbert"]
    val_ids, val_rows, first = kb.validation_image_ids, kb.validation_kb_rows, first_caption_rows(kb)
    own_rows = [np.flatnonzero(kb.caption_image_ids == v) for v in val_ids]
    queries = kb_img[val_rows]
    image_sims, caption_sims = queries @ kb_img.T, queries @ clip_cap.T
    self_top1 = float(np.mean(image_sims.argmax(1) == val_rows))
    checks.add("without exclusion, image->image returns the query itself at rank 1", self_top1 == 1.0,
               rate=self_top1)
    found = (np.array([[h.score for h in hits] for hits in kb.search_images(queries, k=10)]),
             np.array([[h.score for h in hits] for hits in kb.search_captions(queries, k=10, space="clip")]))
    brute = (-np.sort(-image_sims, axis=1)[:, :10], -np.sort(-caption_sims, axis=1)[:, :10])
    difference = max(float(np.abs(f - b).max()) for f, b in zip(found, brute, strict=True))
    checks.add("FAISS top-10 scores equal brute-force NumPy scores", difference < 1e-5,
               max_abs_difference=difference)

    text_to_image = clip_cap[first] @ kb_img.T
    sbert_sims = sbert_cap[first] @ sbert_cap.T
    sbert_sims[np.arange(len(first)), first] = -np.inf
    hits = kb.search_images(queries, k=10, exclude=[{int(v)} for v in val_ids])
    kb_row = {int(image_id): r for r, image_id in enumerate(kb.kb_image_ids)}
    sizes = categories.sum(1)
    jaccard: dict[int, list[float]] = {1: [], 5: [], 10: []}
    baseline = []
    for row, neighbours in zip(val_rows, hits, strict=True):
        if sizes[row] == 0:
            continue
        inter = categories @ categories[row]
        scores = inter / (sizes + sizes[row] - inter)
        baseline.append(float(np.delete(scores, row).mean()))
        rows = [kb_row[h.image_id] for h in neighbours]
        for k in jaccard:
            jaccard[k].append(float(scores[rows[:k]].mean()))
    return {
        "queries": "100 validation images (KB members); vectors are their KB rows",
        "image_to_own_captions_clip": {
            "relevant": "the query image's own KB captions (linkage sanity check; own captions NOT excluded)",
            **recall([rank_of(caption_sims[i], own_rows[i]) for i in range(len(val_ids))])},
        "caption_to_own_image_clip": {
            "relevant": "the image of the query caption (first caption of each validation image)",
            **recall([rank_of(text_to_image[i], np.array([val_rows[i]])) for i in range(len(val_ids))])},
        "caption_to_sibling_captions_sbert": {
            "relevant": "other captions of the same image; the query caption itself is removed",
            **recall([rank_of(sbert_sims[i], own_rows[i][own_rows[i] != first[i]]) for i in range(len(val_ids))])},
        "image_to_image_self_excluded_category_jaccard": {
            "relevant": "mean Jaccard overlap of COCO object-category sets with the top-k neighbours",
            "queries_with_objects": len(baseline),
            **{f"mean_jaccard@{k}": round(float(np.mean(v)), 4) for k, v in jaccard.items()},
            "random_kb_image_baseline": round(float(np.mean(baseline)), 4)},
    }


def median_ms(function: Callable[[Any], Any], inputs: Sequence[Any]) -> float:
    """Median wall-clock milliseconds of one call per input, after one warm-up call."""
    function(inputs[0])
    times = []
    for item in inputs:
        start = time.perf_counter()
        function(item)
        times.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(times)), 3)


def measure_latency(kb: KnowledgeBase, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Single-query latency through the public API (search includes ID mapping)."""
    images = list(arrays["kb_images"][kb.validation_kb_rows])
    paths = [kb.image_path(int(i)) for i in kb.validation_image_ids[:20]]
    texts = kb.caption_texts[:20]
    return {
        "search_image_index_k10": median_ms(lambda q: kb.search_images(q, k=10), images),
        "search_clip_caption_index_k10": median_ms(lambda q: kb.search_captions(q, k=10, space="clip"), images),
        "search_sbert_caption_index_k10": median_ms(lambda q: kb.search_captions(q, k=10, space="sbert"),
                                                    list(arrays["kb_captions_sbert"][:100])),
        "encode_one_image_clip": median_ms(lambda p: kb.encode_images([p]), paths),
        "encode_one_text_clip": median_ms(lambda t: kb.encode_texts([t], "clip"), texts),
        "encode_one_text_sbert": median_ms(lambda t: kb.encode_texts([t], "sbert"), texts),
        "encoder_device": str(kb.encoders.device), "faiss_threads": faiss.omp_get_max_threads(),
    }


def thumbnail(path: Path, size: int = 320) -> np.ndarray:
    """Small RGB copy of an image for figures."""
    image = load_rgb_image(path)
    image.thumbnail((size, size))
    return np.asarray(image)


def show(ax: plt.Axes, path: Path, title: str, color: str) -> None:
    """Draw one thumbnail with a small title and no axes."""
    ax.imshow(thumbnail(path))
    ax.set_title(title, fontsize=8, color=color)
    ax.set_axis_off()


def plot_retrieval_examples(kb: KnowledgeBase, arrays: dict[str, np.ndarray], out: Path) -> Path:
    """Four random validation queries with their top-5 KB images and top KB caption (self excluded)."""
    picks = np.random.default_rng(SEED).choice(len(kb.validation_image_ids), 4, replace=False)
    fig = plt.figure(figsize=(13, 13))
    grid = fig.add_gridspec(8, 6, height_ratios=[1, 0.42] * 4, hspace=0.3, wspace=0.06, top=0.93, bottom=0.02)
    for slot, i in enumerate(picks):
        image_id, query = int(kb.validation_image_ids[i]), arrays["kb_images"][kb.validation_kb_rows[i]]
        neighbours = kb.search_images(query, k=5, exclude=[{image_id}])[0]
        caption = kb.search_captions(query, k=1, space="clip", exclude=[{image_id}])[0][0]
        show(fig.add_subplot(grid[2 * slot, 0]), kb.image_path(image_id), f"Validation query\nimage {image_id}", INK)
        for col, hit in enumerate(neighbours, 1):
            show(fig.add_subplot(grid[2 * slot, col]), kb.image_path(hit.image_id),
                 f"#{col}  cosine {hit.score:.3f}\nimage {hit.image_id}", MUTED)
        text = fig.add_subplot(grid[2 * slot + 1, :])
        text.set_axis_off()
        text.text(0, 0.95, "Own caption: " + textwrap.shorten(kb.captions_for_image(image_id)[0], 150),
                  fontsize=8.5, color=INK, va="top")
        text.text(0, 0.62, f"Top KB caption (CLIP, own captions excluded, cosine {caption.score:.3f}): "
                  + textwrap.shorten(caption.caption or "", 120), fontsize=8.5, color=MUTED, va="top")
    fig.suptitle("Validation queries (random, seed 42): top-5 KB images, query image excluded",
                 fontsize=11, y=0.965)
    path = out / "retrieval_examples.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top_pairs(kb: KnowledgeBase, leakage: dict[str, Any], out: Path) -> Path:
    """The five most similar test->KB and validation->other-KB pairs, for visual review."""
    scan = leakage["near_duplicate_scan"]
    sections = [("Test", scan["test_vs_kb"]["top_pairs"][:5]),
                ("Validation", scan["validation_vs_other_kb"]["top_pairs"][:5])]
    fig, axes = plt.subplots(4, 5, figsize=(13, 9.5), gridspec_kw={"top": 0.91, "bottom": 0.02, "hspace": 0.35})
    for s, (label, pairs) in enumerate(sections):
        for col, pair in enumerate(pairs):
            flag = "  (review)" if pair["cosine"] >= NEAR_DUPLICATE_COSINE else ""
            show(axes[2 * s, col], kb.image_path(pair["query_image_id"]),
                 f"{label} image {pair['query_image_id']}", INK)
            show(axes[2 * s + 1, col], kb.image_path(pair["kb_image_id"]),
                 f"nearest KB image {pair['kb_image_id']}\ncosine {pair['cosine']:.3f}{flag}", MUTED)
    fig.suptitle("Leakage audit: most similar test-to-KB and validation-to-other-KB pairs "
                 f"(CLIP image cosine; review threshold {NEAR_DUPLICATE_COSINE})", fontsize=11, y=0.965)
    path = out / "leakage_top_pairs.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Run all checks, write qc_report.json and figures, exit 1 if a hard check fails."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--embed-dir", type=Path, default=EMBED_DIR)
    parser.add_argument("--device", default="auto", help="auto (default), cuda or cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = read_json(args.embed_dir / "metadata.json")
    kb, checks = KnowledgeBase(args.embed_dir, args.data_dir, args.device), Checks()
    arrays = check_structure(kb, checks, args.data_dir, meta)
    leakage = leakage_audit(kb, checks, args.data_dir, arrays)
    consistency = reencode_consistency(kb, checks, args.data_dir, arrays)
    validation = validation_retrieval(kb, checks, arrays, category_matrix(args.data_dir, kb.kb_image_ids.tolist()))
    latency = measure_latency(kb, arrays)
    figure_dir = args.embed_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    figures = [plot_retrieval_examples(kb, arrays, figure_dir), plot_top_pairs(kb, leakage, figure_dir)]
    report = {
        "release": meta["release"], "metadata_sha256": sha256_file(args.embed_dir / "metadata.json"),
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": checks.passed, "hard_checks": checks.results, "leakage_audit": leakage,
        "reencoding_consistency": consistency, "validation_retrieval": validation, "latency_ms": latency,
        "figures": [p.relative_to(args.embed_dir).as_posix() for p in figures],
        "protocol": "Retrieval metrics use validation queries only; the test split is used only for the leakage audit.",
    }
    write_json(args.embed_dir / "qc_report.json", report)
    logger.info("%d/%d hard checks passed; report: %s", sum(r["passed"] for r in checks.results),
                len(checks.results), args.embed_dir / "qc_report.json")
    sys.exit(0 if checks.passed else 1)


if __name__ == "__main__":
    main()
