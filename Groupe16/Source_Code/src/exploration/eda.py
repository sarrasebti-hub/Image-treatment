"""Descriptive statistics on original captions; no generated-caption evaluation."""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.tokenize import TreebankWordTokenizer

from src.preprocessing.metadata import load_metadata
from src.utils.core import file_hash, path, read_json, stamp, write_json

TOKENIZER = TreebankWordTokenizer()


def word_tokens(text):
    return [word.lower() for word in TOKENIZER.tokenize(text) if any(c.isalpha() for c in word)]


def summary(values):
    values = np.asarray(values)
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "std_population": float(values.std(ddof=0)), "min": int(values.min()), "max": int(values.max())}


def table(cfg, split, name, pairs, labels):
    destination = path(cfg, "tables") / f"{split}_{name}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(labels)
        writer.writerows(pairs)


def save_figure(cfg, split, name, figure):
    filename = path(cfg, "figures") / f"{split}_{name}.png"
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(filename, dpi=160)
    plt.close(figure)
    return filename.name


def bars(cfg, split, name, pairs, title, xlabel="Count", horizontal=False):
    pairs = list(pairs)
    fig, ax = plt.subplots(figsize=(11, max(4, len(pairs) * .23)) if horizontal else (11, 5))
    labels, values = zip(*pairs) if pairs else ([], [])
    if horizontal:
        ax.barh(labels, values)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    else:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel(xlabel)
    ax.set_title(f"{split}: {title}")
    return save_figure(cfg, split, name, fig)


def run_eda(cfg, data=None):
    if data is None:
        # Metadata already carries captions/objects; do not reload the entire train COCO tree.
        from types import SimpleNamespace
        from src.data.coco import image_path
        source = path(cfg, "annotations") / "instances_val2017.json"
        expected_hash = read_json(path(cfg, "metadata") / "manifest.json")["annotation_sha256"][source.name]
        if file_hash(source) != expected_hash:
            raise ValueError("Category annotation inventory changed since metadata creation")
        categories = read_json(source)["categories"]
        data = SimpleNamespace(categories={c["id"]: c for c in categories},
                               image_path=lambda source, row: image_path(cfg, source, row))
    # Local cache only. Missing POS resources fail clearly instead of inventing tags.
    nltk.data.path.insert(0, str(path(cfg, "cache") / "nltk_data"))
    try:
        nltk.pos_tag(["verification"], lang="eng")
    except LookupError as exc:
        raise RuntimeError("POS model missing. Run: python -m src.cli download-nlp") from exc
    results = {}
    views = ["knowledge_base", "validation", "test"]
    comparison_file = path(cfg, "splits") / "comparison/test_100_ids.json"
    if comparison_file.is_file():
        views.append("test_100")
    for split in views:
        records = load_metadata(cfg, "test" if split == "test_100" else split)
        if split == "test_100":
            ids = read_json(comparison_file)
            mapping = {r["image_id"]: r for r in records}
            records = [mapping[i] for i in ids]
        classes, vocabulary, pos, attributes, spatial, scene_counts = (Counter() for _ in range(6))
        classes.update({category["name"]: 0 for category in data.categories.values()})
        lengths, objects, distinct = [], [], []
        for row in records:
            classes.update(obj["category"] for obj in row["objects"])
            objects.append(row["object_count"])
            distinct.append(row["distinct_categories"])
            scene_counts[row["complexity"]["value"]] += 1
            spatial.update(row["spatial_relations"]["counts"])
            for category, terms in row["derived_attributes"]["values"].items():
                attributes.update(f"{category}:{term}" for term in terms)
            for caption in row["captions"]:
                tokens = word_tokens(caption["caption_raw"])
                lengths.append(len(tokens))
                vocabulary.update(tokens)
                # Tag the full case-preserved token sequence; punctuation excluded from counts.
                tags = nltk.pos_tag(TOKENIZER.tokenize(caption["caption_raw"]), lang="eng")
                pos.update(tag for token, tag in tags if any(c.isalpha() for c in token))
        top = classes.most_common(cfg["analysis"]["top_classes"])
        table(cfg, split, "object_classes", classes.most_common(), ["category", "annotation_instances"])
        table(cfg, split, "word_frequencies", vocabulary.most_common(), ["word", "count"])
        table(cfg, split, "pos", pos.most_common(), ["penn_treebank_tag", "count"])
        table(cfg, split, "derived_attributes", attributes.most_common(), ["caption_heuristic", "images_with_keyword"])
        table(cfg, split, "spatial_relations", spatial.items(), ["bbox_heuristic", "ordered_box_pairs"])
        figures = [bars(cfg, split, "top50_objects", top, "Top object classes", "Annotation instances", horizontal=True),
                   bars(cfg, split, "word_frequencies", vocabulary.most_common(30), "Most frequent words", "Word occurrences"),
                   bars(cfg, split, "pos", pos.most_common(), "POS distribution (Penn Treebank)", "Word tokens"),
                   bars(cfg, split, "complexity", scene_counts.items(), "Derived scene complexity", "Images"),
                   bars(cfg, split, "derived_attributes", attributes.most_common(), "Caption keyword mentions (derived, not visual ground truth)", "Images mentioning keyword", horizontal=True),
                   bars(cfg, split, "spatial_relations", spatial.items(), "Box geometry proxies (derived)", "Ordered box pairs")]
        for name, values, xlabel in (("objects_per_image", objects, "Annotation instances per image (includes crowd)"),
                                     ("caption_length", lengths, "Words per original caption")):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(values, bins="auto", edgecolor="white")
            ax.set(xlabel=xlabel, ylabel="Images" if name.startswith("objects") else "Captions", title=f"{split}: {name.replace('_', ' ')}")
            figures.append(save_figure(cfg, split, name, fig))
        labels = [category for category, _ in classes.most_common(cfg["analysis"]["cooccurrence_classes"])]
        cooccurrence = np.zeros((len(labels), len(labels)), dtype=int)
        for row in records:
            present = {obj["category"] for obj in row["objects"]}
            indices = [j for j, label in enumerate(labels) if label in present]
            cooccurrence[np.ix_(indices, indices)] += 1
        table(cfg, split, "cooccurrence", [[label, *cooccurrence[j].tolist()] for j, label in enumerate(labels)], ["category", *labels])
        if labels:
            fig, ax = plt.subplots(figsize=(11, 10))
            plot = ax.imshow(cooccurrence)
            ax.set_xticks(range(len(labels)), labels, rotation=90)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_title(f"{split}: category co-occurrence (images; diagonal = prevalence)")
            fig.colorbar(plot, ax=ax, label="Images")
            figures.append(save_figure(cfg, split, "cooccurrence", fig))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(objects, [np.mean([len(word_tokens(c["caption_raw"])) for c in r["captions"]]) for r in records], s=10, alpha=.35)
        ax.set(xlabel="Annotation instances per image", ylabel="Mean caption length (words)", title=f"{split}: scene complexity and caption length")
        figures.append(save_figure(cfg, split, "complexity_caption_scatter", fig))
        examples = records[:cfg["analysis"]["example_images"]]
        if examples:
            from matplotlib.patches import Rectangle
            fig, axes = plt.subplots(len(examples), 1, figsize=(12, 5 * len(examples)), squeeze=False)
            for row, ax in zip(examples, axes.flat):
                from PIL import Image
                with Image.open(data.image_path(row["coco_source"], row)) as original:
                    ax.imshow(original.convert("RGB"))  # Original coordinates, no EXIF transform.
                for obj in row["objects"]:
                    if not obj.get("bbox_valid", True):
                        continue
                    x, y, w, h = obj["bbox"]
                    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1))
                    ax.text(x, y, obj["category"], color="black", fontsize=7, bbox={"facecolor": "white", "alpha": .7, "pad": 1})
                import textwrap
                caption = "\n".join(textwrap.fill(c["caption_raw"], 100) for c in row["captions"])
                ax.set_title(f"COCO {row['image_id']} | {row['complexity']['value']}")
                ax.set_xlabel(caption, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            figures.append(save_figure(cfg, split, "examples", fig))
        results[split] = {
            "images": len(records), "captions": len(lengths), "captions_per_image": len(lengths) / len(records),
            "objects_per_image": summary(objects), "distinct_categories_per_image": summary(distinct),
            "caption_length_words": summary(lengths), "vocabulary_size": len(vocabulary), "word_tokens": sum(vocabulary.values()),
            "object_classes": dict(classes), "pos_penn_treebank": dict(pos), "complexity": dict(scene_counts),
            "caption_heuristic_image_counts": dict(attributes), "bbox_heuristic_pair_counts": dict(spatial), "figures": figures,
        }
    tagger = nltk.data.find("taggers/" + cfg["analysis"]["pos_resource"])
    result = {**stamp(cfg), "splits": results,
              "tokenization": "NLTK TreebankWordTokenizer; keep tokens containing alphabetic characters; lowercase for vocabulary; POS tags from original case/full tokens, count alphabetic tokens only",
              "pos_model": {"resource": cfg["analysis"]["pos_resource"], "files_sha256": {f.name: file_hash(f) for f in Path(str(tagger)).glob('*.json')}},
              "metadata_manifest_sha256": file_hash(path(cfg, "metadata") / "manifest.json"),
              "limits": ["Counts describe annotations including crowd, not exact physical object counts.",
                         "Caption keyword mentions do not establish image-level ground truth or absence.",
                         "Box-derived relations are geometric proxies, not annotated interactions.",
                         "Test EDA is descriptive; do not tune configuration on test statistics."]}
    write_json(path(cfg, "reports") / "eda.json", result)
    return result
