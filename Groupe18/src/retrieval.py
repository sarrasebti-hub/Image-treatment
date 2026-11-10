"""Train-gallery FAISS cosine retrieval and held-out query smoke/evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np

from .embeddings import combined_index


def build_index(root: Path, config: dict):
    started=time.time()
    matrix, rows = combined_index(root, config)
    train_ix = [i for i, r in enumerate(rows) if r["split"] == config["retrieval"].get("gallery_split", "train")]
    if not train_ix:
        raise RuntimeError("No training image embeddings available for the retrieval gallery")
    gallery = np.asarray(matrix[train_ix], dtype=np.float32)
    if not np.isfinite(gallery).all():
        raise ValueError("Non-finite values found in gallery embeddings")
    norms = np.linalg.norm(gallery, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("Gallery embeddings are not L2-normalized")
    gallery_ids = [rows[i] for i in train_ix]
    dim = gallery.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(gallery)
    dest = root / "vector_database"
    dest.mkdir(parents=True, exist_ok=True)
    index_path = dest / "image_cosine_train.index"
    faiss.write_index(index, str(index_path))
    metadata = {"technology": "FAISS", "faiss_version":faiss.__version__, "index_type": "IndexFlatIP", "metric": "inner_product on L2-normalized vectors (cosine)",
        "dimension": dim, "ntotal": int(index.ntotal), "gallery_split": "train", "gallery_image_ids": [r["image_id"] for r in gallery_ids],
        "embedding_model": config["embedding_models"]["image"], "embedding_model_revision":json.loads((root/config.get("embedding_output_dir","embeddings/image_embeddings")/"train.metadata.json").read_text(encoding="utf-8")).get("model_revision"),
        "self_match_exclusion": "query ID removed after retrieval; gallery contains train only"}
    (dest / "image_cosine_train.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Read back to verify the serialized FAISS index before query testing.
    reread = faiss.read_index(str(index_path))
    if reread.ntotal != len(gallery_ids) or reread.d != dim:
        raise RuntimeError("FAISS index read-back validation failed")
    metadata["indexing_seconds"]=round(time.time()-started,3)
    metadata["index_size_bytes"]=index_path.stat().st_size
    (dest / "image_cosine_train.metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    return index_path, metadata, reread, gallery_ids


def evaluate(root: Path, config: dict, index=None, metadata=None, gallery_rows=None):
    if index is None:
        path = root / "vector_database" / "image_cosine_train.index"
        index = faiss.read_index(str(path))
        metadata = json.loads((root / "vector_database" / "image_cosine_train.metadata.json").read_text(encoding="utf-8"))
        gallery_rows = [{"image_id": int(i), "split": "train"} for i in metadata["gallery_image_ids"]]
    matrix, rows = combined_index(root, config)
    train_ids = {r["image_id"] for r in gallery_rows}
    query_candidates = [i for i, r in enumerate(rows) if r["split"] in ("validation", "test") and r["image_id"] not in train_ids]
    if not query_candidates:
        raise RuntimeError("No held-out validation/test image embeddings available as queries")
    requested = min(int(config["retrieval"].get("query_count", 20)), len(query_candidates))
    # Deterministic evenly spaced queries over held-out rows, no manifest writes.
    chosen = np.linspace(0, len(query_candidates)-1, requested, dtype=int).tolist()
    query_ix = [query_candidates[i] for i in chosen]
    top_k = int(config["retrieval"].get("top_k", 5))
    # Request one extra neighbor in case a query's own ID is present in a future index.
    scores, inds = index.search(np.asarray(matrix[query_ix], dtype=np.float32), min(top_k + 1, index.ntotal))
    results=[]; errors=[]
    id_to_row = {int(r["image_id"]): r for r in gallery_rows}
    for qi, sc, ix in zip(query_ix, scores, inds):
        q = rows[qi]
        hits=[]
        for score, pos in zip(sc, ix):
            if pos < 0: continue
            match = id_to_row[int(metadata["gallery_image_ids"][int(pos)])]
            if int(match["image_id"]) == int(q["image_id"]):
                continue
            hits.append({"image_id": int(match["image_id"]), "score_cosine": float(score)})
            if len(hits) == top_k: break
        if not hits: errors.append(int(q["image_id"]))
        results.append({"query_image_id": int(q["image_id"]), "query_split": q["split"], "neighbors": hits})
    out = root / "reports" / "retrieval"; out.mkdir(parents=True, exist_ok=True)
    (out / "heldout_image_similarity.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    stats={"status": "PASS" if len(results)==requested and not errors else "FAIL", "query_count": len(results), "requested_query_count": requested,
        "top_k": top_k, "successful_queries": len(results)-len(errors), "queries_without_results": errors,
        "gallery_count": index.ntotal, "gallery_split": "train", "query_splits": ["validation", "test"],
        "self_match_count": sum(any(n["image_id"] == r["query_image_id"] for n in r["neighbors"]) for r in results),
        "mean_top1_cosine": float(np.mean([r["neighbors"][0]["score_cosine"] for r in results if r["neighbors"]])) if results else None,
        "index_readback_valid": index.ntotal == metadata["ntotal"]}
    (out / "retrieval_evaluation.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def run(root: Path, config: dict):
    path, meta, index, rows = build_index(root, config)
    result = evaluate(root, config, index, meta, rows)
    result["index_path"] = path.relative_to(root).as_posix()
    return result
