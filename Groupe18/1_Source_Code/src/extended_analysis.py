"""Reproducible vocabulary, visual diversity, geometry and preprocessing audit."""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps, ImageDraw


TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)
COLOR_LABELS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "white", "gray", "black"]


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load(root: Path, rel: str):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def vocabulary(root: Path):
    records = []
    split_counts = {}
    counts = collections.Counter()
    lengths = []
    for split in ("train", "validation", "test"):
        rows = _load(root, f"data/captions/{split}.json")
        split_counts[split] = {"captions": len(rows), "tokens": 0, "unique_words": 0}
        for row in rows:
            raw = row["caption_raw"]
            normalized = unicodedata.normalize("NFKC", raw).casefold()
            tokens = TOKEN_RE.findall(normalized)
            counts.update(tokens)
            lengths.append(len(tokens))
            split_counts[split]["tokens"] += len(tokens)
            records.append((split, row["image_id"], row["caption_id"], len(tokens)))
        split_counts[split]["unique_words"] = len({t for row in rows for t in TOKEN_RE.findall(unicodedata.normalize("NFKC", row["caption_raw"]).casefold())})
    total = sum(counts.values())
    out = root / "reports" / "statistical_analysis"
    _json(out / "caption_vocabulary_statistics.json", {
        "caption_count": len(records), "token_count": total, "vocabulary_size": len(counts),
        "unique_word_count": len(counts), "tokenization": {
            "source": "caption_raw, all frozen splits", "unicode_normalization": "NFKC",
            "case": "casefold", "pattern": r"[^\W_]+(?:['’][^\W_]+)*",
            "description": "Unicode letter/number sequences; underscores excluded; internal straight/curly apostrophes retained; punctuation and whitespace delimit tokens.",
            "stemming_or_lemmatization": False
        },
        "caption_length_tokens": {"mean": statistics.mean(lengths), "median": statistics.median(lengths), "min": min(lengths), "max": max(lengths)},
        "split_statistics": split_counts,
        "top_500_words": [{"rank": i, "word": w, "count": c, "relative_frequency": c / total} for i, (w, c) in enumerate(counts.most_common(500), 1)],
        "limitations": "Vocabulary depends on the stated tokenizer; casefolding merges case variants while preserving inflectional forms."
    })
    pd.DataFrame([{"rank": i, "word": w, "count": c, "relative_frequency": c / total} for i, (w, c) in enumerate(counts.most_common(500), 1)]).to_csv(out / "caption_word_frequencies.csv", index=False)
    pd.DataFrame(records, columns=["split", "image_id", "caption_id", "token_count"]).to_csv(out / "caption_token_counts.csv", index=False)
    import matplotlib.pyplot as plt
    top = counts.most_common(30)
    fig, ax = plt.subplots(figsize=(11, 8)); ax.barh([w for w, _ in top][::-1], [c for _, c in top][::-1]); ax.set(title="Most frequent caption tokens (raw captions)", xlabel="Token count", ylabel="Token"); fig.tight_layout(); fig.savefig(root / "visualizations" / "caption_vocabulary_top30.png", dpi=180); plt.close(fig)
    ranks = np.arange(1, len(counts) + 1)
    freq = np.array(sorted(counts.values(), reverse=True))
    fig, ax = plt.subplots(figsize=(8, 5)); ax.loglog(ranks, freq); ax.set(title="Caption vocabulary rank-frequency curve", xlabel="Token rank (log scale)", ylabel="Frequency (log scale)"); ax.grid(True, which="both", alpha=.2); fig.tight_layout(); fig.savefig(root / "visualizations" / "caption_vocabulary_rank_frequency.png", dpi=180); plt.close(fig)
    return {"vocabulary_size": len(counts), "token_count": total, "caption_count": len(records)}


def _color_metrics(image: Image.Image):
    arr = np.asarray(image.convert("RGB").resize((96, 96), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    delta = mx - mn
    hue = np.zeros_like(mx)
    mask = delta > 1e-6
    mr = (mx == r) & mask; mg = (mx == g) & mask; mb = (mx == b) & mask
    hue[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6
    hue[mg] = (b[mg] - r[mg]) / delta[mg] + 2
    hue[mb] = (r[mb] - g[mb]) / delta[mb] + 4
    hue *= 60
    sat = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0)
    lum = .2126 * r + .7152 * g + .0722 * b
    labels = np.full(hue.shape, "gray", dtype=object)
    vibrant = sat >= .20
    hue_labels = np.array(["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"])
    labels[vibrant] = hue_labels[(hue[vibrant] // 45).astype(int) % 8]
    labels[(~vibrant) & (lum >= .78)] = "white"
    labels[(~vibrant) & (lum <= .20)] = "black"
    palette_counts = collections.Counter(labels.ravel().tolist())
    total = labels.size
    edges = np.asarray(Image.fromarray(np.uint8(lum * 255)).filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
    edge_density = float(np.mean(edges >= 48))
    return {
        "dominant_color_family": palette_counts.most_common(1)[0][0],
        "dominant_color_share": palette_counts.most_common(1)[0][1] / total,
        "color_family_pixel_share": {label: palette_counts.get(label, 0) / total for label in COLOR_LABELS},
        "mean_luminance": float(lum.mean()), "luminance_std": float(lum.std()), "mean_saturation": float(sat.mean()),
        "edge_density_proxy": edge_density,
        "edge_method": "Pillow FIND_EDGES on 96x96 luminance image; fraction of pixels with edge response >=48/255",
        "shape_limitation": "Global edge density is a texture/contour complexity proxy; it does not identify object silhouettes or semantic shape classes."
    }


def visual_diversity(root: Path):
    detail_rows = []
    palette = collections.Counter()
    for split in ("train", "validation", "test"):
        manifest = _load(root, f"data/selected/{split}_ids.json")["image_ids"]
        source_split = "train2017" if split == "train" else "val2017"
        image_ann = _load(root, f"data/raw/annotations/instances_{source_split}.json")["images"]
        filenames = {int(row["id"]): row["file_name"] for row in image_ann}
        source_dir = root / "data" / "raw" / source_split
        for iid in manifest:
            path = source_dir / filenames[int(iid)]
            if not path.exists(): path = source_dir / source_split / filenames[int(iid)]
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                metrics = _color_metrics(im)
            row = {"image_id": int(iid), "split": split, **metrics}
            detail_rows.append(row)
            palette[row["dominant_color_family"]] += 1
    df = pd.DataFrame(detail_rows)
    metadata = root / "data" / "metadata"
    df.to_csv(metadata / "color_shape_diversity.csv", index=False)
    _json(metadata / "color_shape_diversity.json", {"method": "96x96 RGB sampling, HSV-family quantization, luminance moments, Pillow FIND_EDGES threshold proxy", "images": len(detail_rows), "dominant_color_counts": dict(palette), "summary": {k: {"mean": float(df[k].mean()), "median": float(df[k].median()), "std": float(df[k].std(ddof=0))} for k in ("dominant_color_share", "mean_luminance", "luminance_std", "mean_saturation", "edge_density_proxy")}, "color_family_pixel_share_mean": {label: float(np.mean([r["color_family_pixel_share"][label] for r in detail_rows])) for label in COLOR_LABELS}, "limitations": ["Color families are coarse pixel-level estimates, not object-level color labels.", "Black letterbox pixels affect distributions for non-square images.", "Edge density is a coarse image complexity proxy; semantic/object shape classification is not attempted."]})
    import matplotlib.pyplot as plt
    colors = ["#d62728", "#ff7f0e", "#f1c40f", "#2ca02c", "#17becf", "#1f77b4", "#9467bd", "#e377c2", "#f0f0f0", "#888888", "#222222"]
    shares = [np.mean([r["color_family_pixel_share"][label] for r in detail_rows]) for label in COLOR_LABELS]
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(COLOR_LABELS, shares, color=colors); ax.set(title="Mean pixel share by coarse color family", ylabel="Share of image pixels"); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(root / "visualizations" / "color_family_distribution.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(df["edge_density_proxy"], bins=30, color="#3569a8"); ax.set(title="Image edge-density proxy distribution", xlabel="Fraction of strong edge pixels", ylabel="Images"); fig.tight_layout(); fig.savefig(root / "visualizations" / "image_edge_density_distribution.png", dpi=180); plt.close(fig)
    return {"images": len(detail_rows), "dominant_color_counts": dict(palette)}


def _geometry_relations(a, b, width, height):
    ax, ay, aw, ah = map(float, a["bbox"]); bx, by, bw, bh = map(float, b["bbox"])
    acx, acy = ax + aw / 2, ay + ah / 2; bcx, bcy = bx + bw / 2, by + bh / 2
    ix = max(0., min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0., min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; union = aw * ah + bw * bh - inter
    labels = []
    # Centroid ordering supplies a stable image-axis direction even when boxes overlap.
    if (bcx - acx) / max(width, 1) >= .02: labels.append("left_of")
    elif (acx - bcx) / max(width, 1) >= .02: labels.append("right_of")
    if (bcy - acy) / max(height, 1) >= .02: labels.append("above")
    elif (acy - bcy) / max(height, 1) >= .02: labels.append("below")
    eps = 1e-6
    if ax >= bx - eps and ay >= by - eps and ax + aw <= bx + bw + eps and ay + ah <= by + bh + eps and aw * ah < bw * bh - eps: labels.append("inside")
    if bx >= ax - eps and by >= ay - eps and bx + bw <= ax + aw + eps and by + bh <= ay + ah + eps and bw * bh < aw * ah - eps: labels.append("contains")
    iou = inter / union if union > 0 else 0
    if iou >= .10: labels.append("overlapping")
    dist = math.hypot((acx - bcx) / max(width, 1), (acy - bcy) / max(height, 1))
    if dist <= .10: labels.append("near")
    return labels, iou, dist


def spatial_relations(root: Path):
    out = root / "reports" / "statistical_analysis"; md = root / "data" / "metadata"
    relations = collections.Counter(); cat_rel = collections.Counter(); per_image = []
    rel_path = out / "geometric_spatial_relationships.csv"
    out.mkdir(parents=True, exist_ok=True)
    with rel_path.open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "object_a_index", "category_a", "object_b_index", "category_b", "relation", "bbox_iou", "normalized_center_distance", "relation_source"])
        writer.writeheader()
        for split in ("train", "validation", "test"):
            rows = _load(root, f"data/metadata/{split}.json")
            for image in rows:
                objects = image.get("objects", []); counts = collections.Counter()
                width, height = float(image["width"]), float(image["height"])
                for i in range(len(objects)):
                    for j in range(i + 1, len(objects)):
                        labels, iou, dist = _geometry_relations(objects[i], objects[j], width, height)
                        for label in labels:
                            writer.writerow({"split": split, "image_id": image["image_id"], "object_a_index": i, "category_a": objects[i]["name"], "object_b_index": j, "category_b": objects[j]["name"], "relation": label, "bbox_iou": round(iou, 6), "normalized_center_distance": round(dist, 6), "relation_source": "inferred geometry from COCO ground-truth bounding boxes; not semantic relation annotation"})
                            relations[label] += 1; counts[label] += 1
                            cat_rel[(label, objects[i]["category"], objects[j]["category"])] += 1
                per_image.append({"split": split, "image_id": image["image_id"], "object_pairs": len(objects) * (len(objects) - 1) // 2, "relation_counts": dict(counts)})
    pd.DataFrame(per_image).to_json(md / "spatial_relationships_per_image.json", orient="records", force_ascii=False, indent=2)
    top_categories = [{"relation": r, "category_a": a, "category_b": b, "count": n} for (r, a, b), n in cat_rel.most_common(100)]
    _json(out / "geometric_spatial_relationships_summary.json", {"source": "COCO object bounding boxes from selected metadata", "images": len(per_image), "relation_counts": dict(relations), "top_category_pairs": top_categories, "definitions": {"left_of/right_of": "object A/B centroid x ordering with normalized separation >= 0.02 image width", "above/below": "object A/B centroid y ordering with normalized separation >= 0.02 image height", "inside/contains": "strict box containment with unequal areas", "overlapping": "intersection over union >= 0.10", "near": "center-to-center distance <= 0.10 after normalizing x/y by image width/height"}, "semantics": "All relations are geometric inferences, not COCO ground-truth semantic relations. They do not imply actions, support, contact, or functional relations."})
    pd.DataFrame([{"relation": k, "pair_count": v} for k, v in relations.most_common()]).to_csv(out / "geometric_spatial_relationship_frequencies.csv", index=False)
    import matplotlib.pyplot as plt
    items = relations.most_common()
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar([x[0] for x in items], [x[1] for x in items], color="#3569a8"); ax.set(title="Inferred geometric relationships from COCO boxes", ylabel="Directed object-pair count"); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(root / "visualizations" / "geometric_relationship_frequencies.png", dpi=180); plt.close(fig)
    return {"images": len(per_image), "relation_counts": dict(relations)}


def validation_preprocessing(root: Path):
    ids = _load(root, "data/selected/validation_ids.json")["image_ids"]
    ann = _load(root, "data/raw/annotations/instances_val2017.json")
    imgs = {int(x["id"]): x for x in ann["images"]}
    rows = []
    source_base = root / "data" / "raw" / "val2017"
    for iid in ids:
        name = imgs[int(iid)]["file_name"]
        path = source_base / name
        if not path.exists(): path = source_base / "val2017" / name
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            ar = w / h
            fill = min(ar, 1 / ar)
            # Stretch produces 0 distortion only for square inputs; score is absolute log aspect change.
            stretch_distortion = abs(math.log((224 / w) / (224 / h)))
            rows.append({"image_id": int(iid), "source_width": w, "source_height": h, "source_aspect_ratio": ar,
                         "letterbox_content_fraction": fill, "letterbox_distortion": 0.0, "letterbox_crop_fraction": 0.0,
                         "stretch_content_fraction": 1.0, "stretch_distortion": stretch_distortion, "stretch_crop_fraction": 0.0,
                         "center_crop_content_fraction": fill, "center_crop_distortion": 0.0, "center_crop_crop_fraction": 1 - fill})
    df = pd.DataFrame(rows)
    criteria = {
        "letterbox_224": {"mean_content_fraction": float(df.letterbox_content_fraction.mean()), "mean_aspect_distortion_abs_log": 0.0, "mean_crop_fraction": 0.0, "mean_padding_fraction": float((1 - df.letterbox_content_fraction).mean())},
        "stretch_224": {"mean_content_fraction": 1.0, "mean_aspect_distortion_abs_log": float(df.stretch_distortion.mean()), "mean_crop_fraction": 0.0, "mean_padding_fraction": 0.0},
        "center_crop_224": {"mean_content_fraction": float(df.center_crop_content_fraction.mean()), "mean_aspect_distortion_abs_log": 0.0, "mean_crop_fraction": float(df.center_crop_crop_fraction.mean()), "mean_padding_fraction": 0.0}
    }
    df.to_csv(root / "reports" / "statistical_analysis" / "validation_preprocessing_comparison.csv", index=False)
    result = {"status": "COMPARISON_COMPLETE_GEOMETRIC_CRITERIA_ONLY", "validation_images": len(rows), "historical_selection_evidence_found": False, "configurations": criteria,
             "criteria": ["preserve source aspect ratio", "retain the full image field", "minimize artificial padding after those constraints"],
             "decision": "letterbox_224", "rationale": "The letterbox configuration preserves the full image and aspect ratio for every input; center crop discards image area, while direct stretch distorts non-square images. Padding is accepted to satisfy the encoder's square input requirement.",
             "scope_limit": "This new comparison evaluates geometric preprocessing properties only; it does not establish that this configuration maximizes caption/retrieval model performance. No model embeddings or indexes were regenerated.",
             "timestamp_note": "Generated for this audit; no historical experiment is claimed."}
    _json(root / "reports" / "statistical_analysis" / "validation_preprocessing_comparison.json", result)
    # Contact sheet of representative validation images under the three transforms.
    thumb_ids = list(ids[:12]); cell = 224; label_h = 26
    sheet = Image.new("RGB", (cell * 3, (cell + label_h) * len(thumb_ids)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 4), "Letterbox 224", fill="black"); draw.text((cell + 5, 4), "Stretch 224", fill="black"); draw.text((2 * cell + 5, 4), "Center crop 224", fill="black")
    for row_i, iid in enumerate(thumb_ids):
        name = imgs[int(iid)]["file_name"]; path = source_base / name
        if not path.exists(): path = source_base / "val2017" / name
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            lb = Image.new("RGB", (224, 224), (0, 0, 0)); thumb = im.copy(); thumb.thumbnail((224, 224), Image.Resampling.LANCZOS); lb.paste(thumb, ((224-thumb.width)//2, (224-thumb.height)//2))
            stretch = im.resize((224, 224), Image.Resampling.LANCZOS)
            scale = max(224 / im.width, 224 / im.height); enlarged = im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.LANCZOS); left=(enlarged.width-224)//2; top=(enlarged.height-224)//2; crop=enlarged.crop((left, top, left+224, top+224))
            y = row_i * (cell + label_h) + label_h
            for col, pic in enumerate((lb, stretch, crop)): sheet.paste(pic, (col*cell, y))
            draw.text((2, y+cell+2), str(iid), fill="black")
    sheet.save(root / "visualizations" / "validation_preprocessing_comparison.png")
    return result


def run(root: Path):
    summary = {"vocabulary": vocabulary(root), "visual_diversity": visual_diversity(root), "spatial_relationships": spatial_relations(root), "validation_preprocessing": validation_preprocessing(root)}
    _json(root / "reports" / "statistical_analysis" / "extended_analysis_run.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    print(json.dumps(run(args.root), indent=2))


if __name__ == "__main__":
    main()
