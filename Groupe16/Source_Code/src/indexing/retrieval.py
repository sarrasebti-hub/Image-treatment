"""Real image-query retrieval only; no generated captions or final RAG metrics."""
import copy
import csv
import html
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import numpy as np
from tqdm import tqdm

from src.embeddings.encode import Encoder
from src.indexing.faiss_index import load_indexes, search
from src.preprocessing.metadata import load_metadata, load_rgb
from src.utils.core import file_hash, path, stamp, write_json


def frozen_encoder(cfg, manifest):
    settings = copy.deepcopy(cfg)
    settings["embedding"]["revision"] = manifest["resolved_revision"] or cfg["embedding"]["revision"]
    if not manifest["shared_image_text_space"]:
        settings["embedding"]["fallback_text_revision"] = manifest["text_resolved_revision"]
    return Encoder(settings)


def add_evidence(hits, kb, data):
    for hit in hits:
        row = kb[hit["image_id"]]
        hit["file_name"] = row["file_name"]
        hit["coco_source"] = row["coco_source"]
        hit["captions"] = row["captions"]
        hit["metadata_image_id"] = row["image_id"]
    return hits


def write_results(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "results.json", result)
    with (folder / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_image_id", "query_file", "rank", "retrieved_image_id", "faiss_row", "cosine_similarity"])
        writer.writeheader()
        for query in result["queries"]:
            for rank, hit in enumerate(query["results"], 1):
                writer.writerow({"query_image_id": query["image_id"], "query_file": query["file_name"], "rank": rank,
                                 "retrieved_image_id": hit["image_id"], "faiss_row": hit["row"], "cosine_similarity": hit["cosine_similarity"]})


def make_gallery(folder, queries, kb, data, limit=None):
    def image_tag(filename):
        relative = os.path.relpath(filename, folder).replace(os.sep, "/")
        return f'<img loading="lazy" src="{html.escape(quote(relative, safe="/:"))}">'
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>COCO retrieval</title>',
             '<style>body{font-family:system-ui;margin:24px;background:#f6f7f9;color:#172035}section{background:white;padding:18px;margin-bottom:24px;border-radius:12px}.cards{display:flex;flex-wrap:wrap;gap:14px}.card{width:220px}img{width:220px;height:170px;object-fit:contain;background:#eee}p{font-size:13px}h1{font-size:25px}</style>',
             '<h1>CLIP / FAISS image retrieval</h1><p>Retrieved captions are existing knowledge-base references. No captions were generated.</p>']
    for query in queries[:limit] if limit else queries:
        parts += [f'<section><h2>Query {html.escape(str(query["image_id"]))}</h2>', image_tag(query["resolved_path"]), '<div class="cards">']
        for rank, hit in enumerate(query["results"], 1):
            row = kb[hit["image_id"]]
            filename = data.image_path(row["coco_source"], row)
            caption = "<br>".join(html.escape(c["caption_raw"]) for c in row["captions"])
            parts += [f'<div class="card"><h3>#{rank} · COCO {hit["image_id"]}</h3>', image_tag(filename),
                      f'<p>cosine: {hit["cosine_similarity"]:.6f}</p><p>{caption}</p></div>']
        parts += ['</div></section>']
    parts += ['</html>']
    (folder / "gallery.html").write_text("\n".join(parts), encoding="utf-8")


def contact_sheet(folder, filename, image_id, hits, kb, data):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count = len(hits) + 1
    columns = min(4, count)
    figure, axes = plt.subplots(math.ceil(count / columns), columns,
                                figsize=(4 * columns, 4 * math.ceil(count / columns)),
                                squeeze=False, layout="constrained")
    items = [(filename, f"Query: {image_id if image_id is not None else filename.name}")]
    for rank, hit in enumerate(hits, 1):
        row = kb[hit["image_id"]]
        items.append((data.image_path(row["coco_source"], row), f"#{rank} COCO {hit['image_id']}\ncosine = {hit['cosine_similarity']:.4f}"))
    for axis in axes.flat:
        axis.axis("off")
    for axis, (image_file, title) in zip(axes.flat, items):
        with load_rgb(image_file) as image:
            axis.imshow(image)
        axis.set_title(title, fontsize=10)
    figure.suptitle("CLIP / FAISS image retrieval (existing KB images)", fontsize=15)
    figure.savefig(folder / "nearest_images.png", dpi=150)
    plt.close(figure)


def validation_retrieval(cfg, data=None, top_k=None):
    if data is None:
        from src.data.coco import CocoFiles
        data = CocoFiles(cfg)
    top_k = cfg["retrieval"]["top_k"] if top_k is None else top_k
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top-k must be a positive integer")
    indexes, rows, manifest = load_indexes(cfg)
    records = load_metadata(cfg, "validation")
    kb = {r["image_id"]: r for r in load_metadata(cfg, "knowledge_base")}
    encoder = frozen_encoder(cfg, manifest)
    batch_size = cfg["embedding"]["batch_size"]
    vectors, queries, batch_times = [], [], []
    for start in tqdm(range(0, len(records), batch_size), desc="Validation CLIP queries", mininterval=5):
        batch = records[start:start + batch_size]
        images = []
        try:
            for row in batch:
                filename = data.image_path(row["coco_source"], row)
                if file_hash(filename) != row["file_sha256"]:
                    raise ValueError(f"Validation image changed: {row['image_id']}")
                images.append(load_rgb(filename))
            started = perf_counter()
            features = encoder.images(images)
            batch_times.append(perf_counter() - started)
            vectors.append(features)
        finally:
            for image in images:
                image.close()
        for row, feature in zip(batch, features):
            started = perf_counter()
            hits = search(indexes["image"], rows["image"], feature, top_k, row["image_id"])
            latency = (perf_counter() - started) * 1000
            if len(hits) != min(top_k, len(kb)) or any(h["image_id"] not in kb or h["image_id"] == row["image_id"] for h in hits):
                raise ValueError("Invalid validation retrieval result count/provenance")
            queries.append({"image_id": row["image_id"], "file_name": row["file_name"], "split": "validation",
                            "resolved_path": os.path.relpath(data.image_path(row["coco_source"], row), path(cfg, "reports") / "retrieval" / "validation"),
                            "file_sha256": row["file_sha256"], "search_ms": latency,
                            "results": add_evidence(hits, kb, data)})
    folder = path(cfg, "reports") / "retrieval" / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    features = np.concatenate(vectors)
    if features.shape != (cfg["validation_size"], indexes["image"].d) or not np.isfinite(features).all():
        raise ValueError("Validation query embedding count/dimensions invalid")
    np.save(folder / "query_embeddings.npy", features, allow_pickle=False)
    write_json(folder / "query_rows.json", [{"image_id": q["image_id"]} for q in queries])
    result = {**stamp(cfg), "status": "passed", "queries_count": len(queries), "kb_images": len(kb), "top_k": top_k,
              "model": manifest["model"], "resolved_revision": manifest["resolved_revision"],
              "index_manifest_sha256": file_hash(path(cfg, "indexes") / "manifest.json"),
              "query_embeddings_sha256": file_hash(folder / "query_embeddings.npy"),
              "encoding_seconds": sum(batch_times), "mean_search_ms": float(np.mean([q["search_ms"] for q in queries])),
              "purpose": "Retrieval functional validation only; no caption generation or final retrieval relevance metric",
              "queries": queries}
    write_results(folder, result)
    # Resolve image paths only for HTML generation, keep JSON paths relative to this folder.
    gallery_queries = [{**q, "resolved_path": (folder / q["resolved_path"]).resolve()} for q in queries]
    make_gallery(folder, gallery_queries, kb, data)
    return {k: v for k, v in result.items() if k != "queries"} | {"output": str(folder)}


def manual_retrieval(cfg, data=None, image_id=None, image_path=None, top_k=None, output=None, show=False):
    if data is None:
        from src.data.coco import CocoFiles
        data = CocoFiles(cfg)
    if (image_id is None) == (image_path is None):
        raise ValueError("Choose exactly one of --image-id and --image-path")
    top_k = cfg["retrieval"]["top_k"] if top_k is None else top_k
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top-k must be a positive integer")
    indexes, rows, manifest = load_indexes(cfg)
    kb = {r["image_id"]: r for r in load_metadata(cfg, "knowledge_base")}
    expected_query = None
    if image_path is not None:
        filename = Path(image_path)
        if not filename.is_absolute():
            filename = Path(cfg["_root"]) / filename
        filename = filename.resolve()
    else:
        expected_query = kb.get(image_id)
        if expected_query is None:
            for split in ("validation", "test"):
                expected_query = next((r for r in load_metadata(cfg, split) if r["image_id"] == image_id), None)
                if expected_query is not None:
                    break
        if expected_query is not None:
            source, row = expected_query["coco_source"], expected_query
        else:
            from src.utils.core import read_json
            matches = []
            for source in ("train2017", "val2017"):
                inventory = read_json(path(cfg, "annotations") / f"captions_{source}.json")["images"]
                matches.extend((source, row) for row in inventory if row["id"] == image_id)
            if len(matches) != 1:
                raise ValueError("COCO ID not found uniquely in local annotation inventory")
            source, row = matches[0]
        filename = data.image_path(source, row)
    if not filename.is_file():
        raise FileNotFoundError(filename)
    query_sha256 = file_hash(filename)
    if expected_query is not None and query_sha256 != expected_query["file_sha256"]:
        raise ValueError("Selected query image changed since metadata creation")
    # A copied KB image queried by path must not retrieve itself either.
    identical_ids = {r["image_id"] for r in kb.values() if r["file_sha256"] == query_sha256}
    encoder = frozen_encoder(cfg, manifest)
    started = perf_counter()
    with load_rgb(filename) as image:
        feature = encoder.images([image])
    encoding_seconds = perf_counter() - started
    started = perf_counter()
    hits = search(indexes["image"], rows["image"], feature, top_k + len(identical_ids), image_id)
    hits = [h for h in hits if h["image_id"] not in identical_ids][:top_k]
    latency = (perf_counter() - started) * 1000
    add_evidence(hits, kb, data)
    folder = Path(output) if output else path(cfg, "reports") / "retrieval" / f"manual_{image_id if image_id is not None else query_sha256[:12]}_k{top_k}"
    if not folder.is_absolute():
        folder = Path(cfg["_root"]) / folder
    query = {"image_id": image_id, "file_name": filename.name, "resolved_path": os.path.relpath(filename, folder),
             "file_sha256": query_sha256, "encoding_seconds": encoding_seconds, "search_ms": latency,
             "excluded_identical_file_ids": sorted(identical_ids), "results": hits}
    result = {**stamp(cfg), "model": manifest["model"], "resolved_revision": manifest["resolved_revision"],
              "top_k": top_k, "kb_images": len(kb), "queries": [query]}
    write_results(folder, result)
    make_gallery(folder, [{**query, "resolved_path": filename}], kb, data)
    contact_sheet(folder, filename, image_id, hits, kb, data)
    if show:
        import webbrowser
        webbrowser.open((folder / "gallery.html").resolve().as_uri())
    return {"query_image_id": image_id, "top_k": top_k, "results": [{k: hit[k] for k in ("image_id", "row", "cosine_similarity")} for hit in hits],
            "output": str(folder), "gallery": str(folder / "gallery.html")}
