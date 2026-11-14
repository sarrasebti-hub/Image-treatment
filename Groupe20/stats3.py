import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG, REP = Path("figures"), Path("reports")
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)
results = {}

det_val = json.load(open("metadata/detected_val.json"))["images"]
gt_val = json.load(open("metadata/val_objects_gt.json"))["images"]
det_kb = json.load(open("metadata/detected_kb.json"))["images"]
gt_kb = json.load(open("metadata/kb_metadata.json"))["images"]

# ---------- 1. choose the confidence threshold on the VALIDATION photos ----------
# for each photo, compare the set of classes found by the detector with the human labels
table = []
for thr in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    tp = fp = fn = 0
    for i, gt in gt_val.items():
        g = set(gt["category_counts"])
        p = {o["category"] for o in det_val[i]["objects"] if o["confidence"] >= thr}
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    table.append({"threshold": thr, "precision": round(prec, 3),
                  "recall": round(rec, 3), "f1": round(f1, 3)})
    print(f"threshold {thr:.2f}: precision {prec:.3f} recall {rec:.3f} F1 {f1:.3f}")

best = max(table, key=lambda r: r["f1"])
THR = best["threshold"]
print("\nSelected threshold (best F1 on validation):", THR)
results["threshold_selection"] = {"table": table, "selected": THR,
                                  "method": "class-presence F1 vs COCO human labels, 100 validation photos"}

plt.figure(figsize=(6, 4))
plt.plot([r["threshold"] for r in table], [r["precision"] for r in table], marker="o", label="precision")
plt.plot([r["threshold"] for r in table], [r["recall"] for r in table], marker="o", label="recall")
plt.plot([r["threshold"] for r in table], [r["f1"] for r in table], marker="o", label="F1")
plt.axvline(THR, color="grey", linestyle="--")
plt.xlabel("confidence threshold"); plt.legend(); plt.title("Threshold selection (validation)")
plt.tight_layout(); plt.savefig(FIG / "threshold_selection.png", dpi=150); plt.close()

# ---------- 2. exploration of the detected objects (knowledge base) ----------
kept = {i: [o for o in v["objects"] if o["confidence"] >= THR] for i, v in det_kb.items()}
det_classes = Counter(o["category"] for objs in kept.values() for o in objs)
det_counts = [len(objs) for objs in kept.values()]
gt_classes = Counter()
for v in gt_kb.values():
    gt_classes.update(v["category_counts"])
gt_counts = [v["n_objects"] for v in gt_kb.values()]

results["detected_objects"] = {
    "threshold": THR,
    "avg_objects_per_image": round(float(np.mean(det_counts)), 2),
    "images_with_no_objects": int(sum(c == 0 for c in det_counts)),
    "top10_classes": det_classes.most_common(10),
}
results["human_labels"] = {
    "avg_objects_per_image": round(float(np.mean(gt_counts)), 2),
    "images_with_no_objects": int(sum(c == 0 for c in gt_counts)),
}
print("\nDetected @", THR, ":", results["detected_objects"]["avg_objects_per_image"],
      "objects/image | human labels:", results["human_labels"]["avg_objects_per_image"])

top = det_classes.most_common(50)
plt.figure(figsize=(8, 10))
plt.barh([n for n, _ in top][::-1], [c for _, c in top][::-1])
plt.title(f"Top 50 detected classes (KB, conf >= {THR})")
plt.xlabel("Number of detections")
plt.tight_layout(); plt.savefig(FIG / "top50_classes_detected.png", dpi=150); plt.close()

plt.figure()
bins = range(0, 40)
plt.hist(gt_counts, bins=bins, alpha=0.6, label="human labels (COCO)")
plt.hist(det_counts, bins=bins, alpha=0.6, label=f"detector (conf >= {THR})")
plt.legend(); plt.xlabel("objects per image"); plt.ylabel("number of images")
plt.title("Objects per image: humans vs detector")
plt.savefig(FIG / "objects_per_image_detected_vs_human.png", dpi=150); plt.close()

top15 = [c for c, _ in gt_classes.most_common(15)]
x = np.arange(len(top15))
plt.figure(figsize=(10, 4))
plt.bar(x - 0.2, [gt_classes[c] for c in top15], 0.4, label="human labels")
plt.bar(x + 0.2, [det_classes[c] for c in top15], 0.4, label="detector")
plt.xticks(x, top15, rotation=60, ha="right"); plt.legend()
plt.title("Instances per class, top 15 human-labelled classes")
plt.tight_layout(); plt.savefig(FIG / "classes_detected_vs_human.png", dpi=150); plt.close()

# co-occurrence of detected classes
img_count, pairs = Counter(), Counter()
for objs in kept.values():
    cats = sorted({o["category"] for o in objs})
    img_count.update(cats)
    pairs.update(combinations(cats, 2))
names = [c for c, _ in img_count.most_common(25)]
M = np.zeros((25, 25))
for a, ca in enumerate(names):
    M[a, a] = img_count[ca]
    for b, cb in enumerate(names):
        if a != b:
            M[a, b] = pairs[tuple(sorted((ca, cb)))]
plt.figure(figsize=(9, 8))
plt.imshow(np.log1p(M), cmap="viridis")
plt.xticks(range(25), names, rotation=90, fontsize=7); plt.yticks(range(25), names, fontsize=7)
plt.colorbar(label="log(1 + number of images)")
plt.title("Co-occurrence of detected classes (KB)")
plt.tight_layout(); plt.savefig(FIG / "cooccurrence_detected.png", dpi=150); plt.close()

# ---------- 3. captions: before vs after preprocessing ----------
kb = json.load(open("processed/kb_captions.json"))
raw = [c for r in kb for c in r["caption_raw"]]
clean = [c for r in kb for c in r["caption_clean"]]
for name, caps in [("before_preprocessing", raw), ("after_preprocessing", clean)]:
    toks = [c.split() for c in caps]
    results[f"captions_{name}"] = {
        "avg_length_words": round(float(np.mean([len(t) for t in toks])), 2),
        "vocabulary_size": len({w for t in toks for w in t}),
    }
    print(name, results[f"captions_{name}"])

json.dump(results, open(REP / "dataset_stats_detected.json", "w"), indent=1)
print("\nSaved reports/dataset_stats_detected.json and 5 figures in figures/")