import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANN_DIR = Path(r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

kb_ids = set(json.load(open("splits/kb_train_ids.json")))

# ---------- captions of the knowledge base ----------
caps = defaultdict(list)
for a in json.load(open(ANN_DIR / "captions_train2017.json"))["annotations"]:
    if a["image_id"] in kb_ids:
        caps[a["image_id"]].append(a["caption"])

all_caps = [c for i in kb_ids for c in caps[i][:5]]      # keep 5 per image
tokens = [re.findall(r"[a-z']+", c.lower()) for c in all_caps]
lengths = [len(t) for t in tokens]
vocab = {w for t in tokens for w in t}

print("Images:", len(kb_ids), "| Captions:", len(all_caps))
print("Avg caption length (words):", round(np.mean(lengths), 2))
print("Vocabulary size:", len(vocab))

# ---------- objects of the knowledge base ----------
inst = json.load(open(ANN_DIR / "instances_train2017.json"))
cat_name = {c["id"]: c["name"] for c in inst["categories"]}

obj_per_img = Counter()
class_count = Counter()
for a in inst["annotations"]:
    if a["image_id"] in kb_ids:
        obj_per_img[a["image_id"]] += 1
        class_count[cat_name[a["category_id"]]] += 1

counts = [obj_per_img[i] for i in kb_ids]                # 0 if no objects
print("Avg objects per image:", round(np.mean(counts), 2))
print("Images with 0 objects:", sum(c == 0 for c in counts))

# ---------- plots ----------
top = class_count.most_common(50)
plt.figure(figsize=(8, 10))
plt.barh([n for n, _ in top][::-1], [c for _, c in top][::-1])
plt.title("Top 50 object classes (knowledge base)")
plt.xlabel("Number of instances")
plt.tight_layout()
plt.savefig(FIG / "top50_classes.png", dpi=150)
plt.close()

plt.figure()
plt.hist(counts, bins=range(0, 40), edgecolor="black")
plt.title("Objects per image")
plt.xlabel("Number of objects")
plt.ylabel("Number of images")
plt.savefig(FIG / "objects_per_image.png", dpi=150)
plt.close()

plt.figure()
plt.hist(lengths, bins=range(0, 40), edgecolor="black")
plt.title("Caption length")
plt.xlabel("Words per caption")
plt.ylabel("Number of captions")
plt.savefig(FIG / "caption_length.png", dpi=150)
plt.close()

print("Plots saved in figures/")