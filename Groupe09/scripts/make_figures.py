"""Regenerate EDA figures from results/eda/*.csv  ->  outputs/figures/ (run from repo root)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
E_, O = Path("results/eda"), Path("outputs/figures"); O.mkdir(parents=True, exist_ok=True)
C = "#4C78A8"
def hbar(df, x, y, title, fn, xlabel, n=None):
    d = df.head(n) if n else df
    fig, ax = plt.subplots(figsize=(8, 0.28*len(d)+1.5))
    ax.barh(d[x][::-1], d[y][::-1], color=C)
    ax.set_title(title); ax.set_xlabel(xlabel)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(O/fn, dpi=200); plt.close(fig)
hbar(pd.read_csv(E_/"top_50_object_categories.csv"), "category_name", "instance_count",
     "Top 50 object categories (KB, 10,000 images)", "top_50_object_categories.png", "Labelled instances")
hbar(pd.read_csv(E_/"colors_caption_cues.csv"), "cue", "caption_count",
     "Colour words in captions (of 50,000)", "colors_caption_cues.png", "Captions")
pos = pd.read_csv(E_/"pos_distribution.csv"); pos = pos[pos.Percentage >= 0.3]
hbar(pos, "POS", "Percentage", "Part-of-speech distribution (%)", "pos_distribution.png", "% of tokens")
oc = pd.read_csv(E_/"object_count_distribution.csv")
fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(oc.object_count.astype(str), oc.image_count, color=C)
for i, v in enumerate(oc.image_count): ax.text(i, v+30, f"{v:,}", ha="center", fontsize=8)
ax.set_title("Labelled objects per image (KB)"); ax.set_xlabel("Objects"); ax.set_ylabel("Images")
fig.tight_layout(); fig.savefig(O/"objects_per_image_distribution.png", dpi=200); plt.close(fig)
pr = pd.read_csv(E_/"top_20_object_pairs.csv"); pr["pair"] = pr.object_a + " + " + pr.object_b
hbar(pr, "pair", "cooccurrence_count", "Top 20 co-occurring object pairs", "top_20_object_pairs.png", "Images containing both")
ctx = pd.read_csv(E_/"contexts_caption_cues.csv")
fig, ax = plt.subplots(figsize=(6, 3.5)); ax.bar(ctx.cue, ctx.caption_count, color=C)
ax.set_title("Context cues in captions (keyword heuristic)"); ax.set_ylabel("Captions")
fig.tight_layout(); fig.savefig(O/"contexts_caption_cues.png", dpi=200); plt.close(fig)
sp = pd.read_csv(E_/"spatial_bbox_proxy_counts.csv"); sp = sp[sp.proxy != "pair_count"]
fig, ax = plt.subplots(figsize=(7, 3.5)); ax.bar(sp.proxy, sp["count"], color=C)
ax.tick_params(axis="x", labelrotation=15); ax.set_title("Bounding-box geometry proxies (not semantic relations)")
fig.tight_layout(); fig.savefig(O/"spatial_bbox_proxies.png", dpi=200); plt.close(fig)
print("figures written to", O)
