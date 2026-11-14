import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np

FIG = Path("figures")
REP = Path("reports")
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

kb = json.load(open("processed/kb_captions.json"))
meta = json.load(open("metadata/kb_metadata.json"))["images"]
results = {}

# ---------------- 1. part-of-speech distribution ----------------
for pkg in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)


def group(tag):
    if tag.startswith("NN"): return "noun"
    if tag.startswith("VB"): return "verb"
    if tag.startswith("JJ"): return "adjective"
    if tag.startswith("RB"): return "adverb"
    if tag == "IN": return "preposition"
    if tag in ("DT", "PDT"): return "determiner"
    if tag == "CD": return "number"
    if tag in ("CC", "TO", "PRP", "PRP$", "WDT", "WP", "WRB", "MD", "RP"): return "function word"
    return "other"


pos = Counter()
for row in kb:
    for cap in row["caption_raw"]:
        tokens = re.findall(r"[A-Za-z']+", cap)
        pos.update(group(t) for _, t in nltk.pos_tag(tokens))
total = sum(pos.values())
results["pos_distribution_percent"] = {k: round(100 * v / total, 2) for k, v in pos.most_common()}
print("POS:", results["pos_distribution_percent"])

plt.figure(figsize=(8, 4))
plt.bar(list(pos.keys()), [100 * v / total for v in pos.values()])
plt.title("Part-of-speech distribution (KB captions)")
plt.ylabel("% of tokens")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIG / "pos_distribution.png", dpi=150)
plt.close()

# ---------------- 2. object co-occurrence matrix ----------------
class_images = Counter()
pairs = Counter()
for info in meta.values():
    cats = sorted(info["category_counts"])
    class_images.update(cats)
    pairs.update(combinations(cats, 2))

top = [c for c, _ in class_images.most_common(25)]
M = np.zeros((len(top), len(top)))
for i, a in enumerate(top):
    M[i, i] = class_images[a]
    for j, b in enumerate(top):
        if i != j:
            M[i, j] = pairs[tuple(sorted((a, b)))]

plt.figure(figsize=(9, 8))
plt.imshow(np.log1p(M), cmap="viridis")
plt.xticks(range(len(top)), top, rotation=90, fontsize=7)
plt.yticks(range(len(top)), top, fontsize=7)
plt.colorbar(label="log(1 + number of images)")
plt.title("Co-occurrence of the 25 most frequent classes (KB)")
plt.tight_layout()
plt.savefig(FIG / "cooccurrence_matrix.png", dpi=150)
plt.close()
results["top_cooccurring_pairs"] = [[list(p), n] for p, n in pairs.most_common(10)]

# ---------------- 3. diversity (keyword based) ----------------
COLORS = ["white", "black", "red", "blue", "green", "yellow", "brown", "orange",
          "pink", "purple", "gray", "grey", "silver", "gold"]
SHAPES = ["round", "square", "rectangular", "circular", "oval", "triangular", "tall", "long"]
INDOOR = ["kitchen", "bathroom", "bedroom", "living room", "room", "indoor", "inside",
          "office", "restaurant", "toilet", "sink", "couch", "bed"]
OUTDOOR = ["street", "field", "beach", "park", "road", "grass", "sky", "outdoor",
           "outside", "mountain", "snow", "ocean", "city", "river", "lake", "sidewalk"]
NIGHT = ["night", "dark", "evening", "sunset", "nighttime"]
DAY = ["day", "daytime", "sunny", "morning", "afternoon", "sunlight", "bright"]
SPATIAL = ["on top of", "next to", "in front of", "behind", "under", "above", "below",
           "near", "beside", "between", "inside", "on", "in", "at", "over", "against"]

texts = [" ".join(row["caption_clean"]) for row in kb]          # all 5 captions of an image
all_caps = [c for row in kb for c in row["caption_clean"]]


def has(text, words):
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def count_caps(words):
    return {w: sum(bool(re.search(rf"\b{re.escape(w)}\b", c)) for c in all_caps) for w in words}


results["color_mentions"] = count_caps(COLORS)
results["shape_mentions"] = count_caps(SHAPES)
results["spatial_mentions"] = count_caps(SPATIAL)
results["indoor_images"] = sum(has(t, INDOOR) for t in texts)
results["outdoor_images"] = sum(has(t, OUTDOOR) for t in texts)
results["night_images"] = sum(has(t, NIGHT) for t in texts)
results["day_images"] = sum(has(t, DAY) for t in texts)
print({k: v for k, v in results.items() if k.endswith("_images")})


def bar(d, title, name, horizontal=False):
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(8, 4))
    plt.bar([k for k, _ in items], [v for _, v in items])
    plt.title(title)
    plt.ylabel("captions mentioning it")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()


bar(results["color_mentions"], "Colour words in KB captions", "diversity_colors.png")
bar(results["shape_mentions"], "Shape words in KB captions", "diversity_shapes.png")
bar(results["spatial_mentions"], "Spatial relation words in KB captions", "spatial_relations.png")

plt.figure(figsize=(6, 4))
labels = ["indoor", "outdoor", "day", "night"]
vals = [results[f"{k}_images"] for k in labels]
plt.bar(labels, vals)
plt.title("Scene context (images with a matching keyword)")
plt.ylabel("number of images (out of 10,000)")
plt.tight_layout()
plt.savefig(FIG / "diversity_context.png", dpi=150)
plt.close()

json.dump(results, open(REP / "dataset_stats.json", "w"), indent=1)
print("Saved reports/dataset_stats.json and 6 figures in figures/")