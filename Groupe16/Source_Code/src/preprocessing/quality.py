from src.data.audit import audit
from src.data.splits import load_splits
from src.embeddings.encode import check_matrix, load_embeddings
from src.indexing.faiss_index import load_indexes, search
from src.preprocessing.metadata import clean_caption, load_metadata
from src.utils.core import digest, file_hash, path, read_json, stamp, write_json


def quality_control(cfg, data):
    checks = []
    try:
        audit(cfg, data)
        checks.append("data audit, file/pixel hashes, split provenance, no overlap")
        splits = load_splits(cfg)
        for split in splits:
            rows = load_metadata(cfg, split)
            for row in rows:
                official = {c["id"]: c["caption"] for c in data.references(row["coco_source"], row["image_id"])}
                if len(row["captions"]) != len(official):
                    raise ValueError("Metadata missing captions")
                for caption in row["captions"]:
                    if caption["caption_raw"] != official[caption["caption_id"]]:
                        raise ValueError("Raw caption changed")
                    if caption["caption_clean"] != clean_caption(caption["caption_raw"], cfg["preprocessing"]["lowercase"]):
                        raise ValueError("Clean caption differs from configured transform")
                    if caption.get("source") != "coco_human_reference" or caption.get("confidence") != 1.0:
                        raise ValueError("Human-caption confidence/source policy missing")
                if "confidence" in row["derived_attributes"]:
                    raise ValueError("Derived heuristics must not inherit the human-reference confidence")
        checks.append("metadata IDs, captions, unchanged raw references")
        checks.append("human-reference confidence=1; derived attributes do not inherit human confidence")
        arrays, rows, _ = load_embeddings(cfg)
        checks.append("finite float32 normalized embeddings, dimensions, row identity, input/config hashes")
        indexes, mappings, _ = load_indexes(cfg)
        checks.append("FAISS counts, metric, mapping, reconstructed vector equality")
        for kind in ("image", "text"):
            for i in range(min(3, len(rows[kind]))):
                query_id = rows[kind][i]["image_id"]
                hits = search(indexes[kind], mappings[kind], arrays[kind][i], cfg["retrieval"]["top_k"], query_id)
                if any(h["image_id"] == query_id for h in hits):
                    raise ValueError("Self-retrieval leakage")
                if any(h["image_id"] not in set(splits["knowledge_base"]) for h in hits):
                    raise ValueError("Non-KB image retrieved")
        checks.append("same-image exclusion for image and all text rows")
        retrieval_folder = path(cfg, "reports") / "retrieval" / "validation"
        if (retrieval_folder / "results.json").is_file():
            import numpy as np
            result = read_json(retrieval_folder / "results.json")
            if result["index_manifest_sha256"] != file_hash(path(cfg, "indexes") / "manifest.json"):
                raise ValueError("Validation retrieval is stale: different FAISS release")
            query_rows = read_json(retrieval_folder / "query_rows.json")
            if [r["image_id"] for r in query_rows] != splits["validation"]:
                raise ValueError("Validation query embedding mapping mismatch")
            vectors = np.load(retrieval_folder / "query_embeddings.npy", allow_pickle=False)
            check_matrix(vectors, query_rows, indexes["image"].d)
            if file_hash(retrieval_folder / "query_embeddings.npy") != result["query_embeddings_sha256"]:
                raise ValueError("Saved validation embeddings changed")
            if [q["image_id"] for q in result["queries"]] != splits["validation"]:
                raise ValueError("Validation results missing/misordered queries")
            for query, vector in zip(result["queries"], vectors):
                expected = search(indexes["image"], mappings["image"], vector, result["top_k"], query["image_id"])
                actual = [{k: h[k] for k in ("image_id", "row", "cosine_similarity")} for h in query["results"]]
                if len(actual) != len(expected) or any(
                        a["image_id"] != e["image_id"] or a["row"] != e["row"]
                        or not np.isclose(a["cosine_similarity"], e["cosine_similarity"], atol=1e-6, rtol=1e-6)
                        for a, e in zip(actual, expected)):
                    raise ValueError("Saved validation hits differ from a fresh exact FAISS search")
            checks.append("validation query vectors, IDs, hashes and every saved top-k result replayed against FAISS")
        result = {**stamp(cfg), "passed": True, "checks": checks, "split_sha256": digest(splits)}
    except Exception as exc:
        result = {**stamp(cfg), "passed": False, "checks_passed_before_failure": checks, "error": str(exc)}
        write_json(path(cfg, "reports") / "quality_control.json", result)
        raise
    write_json(path(cfg, "reports") / "quality_control.json", result)
    return result
