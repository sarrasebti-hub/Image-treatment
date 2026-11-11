"""ÉTAPE 3 — Exploration statistique de la base de connaissances (KB).

Entrées : data/splits/kb_ids.json + annotations (cache slim)
Sorties : results/visualizations/eda_*.png, results/reports/eda_stats.json
"""
import itertools
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")                      # pas de fenêtre : on sauvegarde en PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C
from common import caption_attributes, clean_caption, get_slim, load_json, save_json, spatial_relation

sns.set_theme(style="whitegrid")
POS_SAMPLE = 10_000        # nb de captions analysées pour le POS (rapidité)
SPATIAL_SAMPLE = 2_000     # nb d'images pour les relations spatiales


def save(name):
    plt.tight_layout()
    plt.savefig(C.VIZ / name, dpi=150)
    plt.close()
    print("  figure ->", name)


def barh(labels, values, title, xlabel, name, height=5):
    plt.figure(figsize=(8, height))
    sns.barplot(x=list(values), y=list(labels), orient="h", color="steelblue")
    plt.title(title)
    plt.xlabel(xlabel)
    save(name)


def main():
    kb = load_json(C.SPLITS / "kb_ids.json")
    slim = get_slim("train", C.TRAIN_IMG_DIR)
    cat_name = {k: v["name"] for k, v in slim["categories"].items()}
    imgs = {i: slim["images"][i] for i in kb}
    rng = random.Random(C.SEED)
    stats = {"n_images": len(imgs)}

    # ---- 1) Fréquence des classes d'objets (top 50) ----
    cat_count = Counter(o["category_id"] for v in imgs.values() for o in v["objects"])
    top50 = cat_count.most_common(50)
    barh([cat_name[k] for k, _ in top50], [c for _, c in top50],
         "Top 50 des classes d'objets (KB)", "Nombre d'instances", "eda_top50_classes.png", height=11)
    stats["n_categories_present"] = len(cat_count)
    stats["top10_categories"] = {cat_name[k]: c for k, c in top50[:10]}
    stats["category_counts_all"] = {cat_name[k]: c for k, c in cat_count.most_common()}

    # ---- 2) Objets par image + complexité de scène ----
    n_obj = np.array([len(v["objects"]) for v in imgs.values()])
    n_cat = np.array([len({o["category_id"] for o in v["objects"]}) for v in imgs.values()])
    plt.figure(figsize=(8, 4))
    plt.hist(np.clip(n_obj, 0, 30), bins=np.arange(0, 32) - 0.5, color="steelblue")
    plt.title("Objets par image (valeurs > 30 regroupées)")
    plt.xlabel("Nombre d'objets")
    plt.ylabel("Nombre d'images")
    save("eda_objects_per_image.png")
    levels = pd.cut(n_obj, [-1, 3, 8, 10 ** 6], labels=["simple (0-3)", "moyenne (4-8)", "complexe (9+)"])
    lv = levels.value_counts().reindex(levels.categories)
    barh(lv.index, lv.values, "Complexité des scènes (nb d'objets)", "Nombre d'images",
         "eda_scene_complexity.png", height=3)
    stats["objects_per_image"] = {"mean": float(n_obj.mean()), "median": float(np.median(n_obj)),
                                  "min": int(n_obj.min()), "max": int(n_obj.max()),
                                  "images_with_0_objects": int((n_obj == 0).sum())}
    stats["categories_per_image_mean"] = float(n_cat.mean())
    stats["scene_complexity"] = {k: int(v) for k, v in lv.items()}

    # ---- 3) Statistiques textuelles des captions ----
    raw = {i: v["captions"][:C.N_CAPTIONS] for i, v in imgs.items()}
    clean = {i: [clean_caption(c) for c in cs] for i, cs in raw.items()}
    all_clean = [c for cs in clean.values() for c in cs]
    lens = np.array([len(c.split()) for c in all_clean])
    vocab = Counter(w for c in all_clean for w in c.split())
    plt.figure(figsize=(8, 4))
    plt.hist(lens, bins=range(0, int(min(lens.max(), 40)) + 2), color="seagreen")
    plt.title("Longueur des captions (mots)")
    plt.xlabel("Nombre de mots")
    plt.ylabel("Nombre de captions")
    save("eda_caption_length.png")
    stats["captions"] = {"n_captions": len(all_clean), "length_mean": float(lens.mean()),
                         "length_std": float(lens.std()), "length_min": int(lens.min()),
                         "length_max": int(lens.max()), "vocab_size": len(vocab),
                         "top20_words": dict(vocab.most_common(20))}

    # ---- 4) Distribution POS (spaCy) ----
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        sample = rng.sample(all_clean, min(POS_SAMPLE, len(all_clean)))
        pos = Counter()
        with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler"]):
            for doc in nlp.pipe(sample, batch_size=512):
                pos.update(t.pos_ for t in doc)
        tot = sum(pos.values())
        items = pos.most_common()
        barh([k for k, _ in items], [100 * v / tot for _, v in items],
             f"Distribution POS ({len(sample)} captions)", "% des tokens",
             "eda_pos_distribution.png", height=5)
        stats["pos_distribution_pct"] = {k: round(100 * v / tot, 2) for k, v in items}
    except Exception as e:
        print("  POS ignoré (spaCy / modèle manquant ?) :", repr(e)[:120])
        print("  -> pip install spacy && python -m spacy download en_core_web_sm")
        stats["pos_distribution_pct"] = None

    # ---- 5) Relations spatiales (calculées depuis les bounding boxes) ----
    rel = Counter()
    for v in rng.sample(list(imgs.values()), min(SPATIAL_SAMPLE, len(imgs))):
        objs = sorted([o for o in v["objects"] if not o["iscrowd"] and o["area"] > 0],
                      key=lambda o: -o["area"])[:8]
        for a, b in itertools.combinations(objs, 2):
            rel[spatial_relation(a["bbox"], b["bbox"])] += 1
    items = rel.most_common()
    if items:
        barh([k for k, _ in items], [v for _, v in items],
             "Relations spatiales entre objets (heuristique bbox)", "Nombre de paires",
             "eda_spatial_relations.png", height=4)
    stats["spatial_relations"] = dict(rel)

    # ---- 6) Diversité : couleurs / formes / intérieur-extérieur / jour-nuit ----
    attrs = [caption_attributes(clean[i]) for i in imgs]
    col = Counter(c for a in attrs for c in a["colors"])
    shp = Counter(s for a in attrs for s in a["shapes"])
    setting = Counter(a["setting"] for a in attrs)
    tod = Counter(a["time"] for a in attrs)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a_, cnt, t in [(ax[0, 0], col, "Couleurs citées (nb d'images)"),
                       (ax[0, 1], shp, "Formes citées (nb d'images)"),
                       (ax[1, 0], setting, "Intérieur / extérieur"),
                       (ax[1, 1], tod, "Jour / nuit")]:
        items = cnt.most_common()
        if items:
            sns.barplot(x=[v for _, v in items], y=[k for k, _ in items], orient="h",
                        color="darkorange", ax=a_)
        a_.set_title(t)
    save("eda_diversity.png")
    stats["diversity"] = {"colors": dict(col), "shapes": dict(shp),
                          "setting": dict(setting), "time_of_day": dict(tod),
                          "note": "heuristique par mots-clés sur les captions"}

    # ---- 7) Matrice de co-occurrence (top 20 classes) ----
    top20 = [k for k, _ in cat_count.most_common(20)]
    P = np.array([[any(o["category_id"] == c for o in v["objects"]) for c in top20]
                  for v in imgs.values()], dtype=float)
    co = P.T @ P
    plt.figure(figsize=(10, 8))
    sns.heatmap(co, mask=np.eye(len(top20), dtype=bool), cmap="viridis",
                xticklabels=[cat_name[c] for c in top20], yticklabels=[cat_name[c] for c in top20])
    plt.title("Co-occurrence des 20 classes les plus fréquentes (nb d'images, diagonale masquée)")
    save("eda_cooccurrence.png")

    # ---- 8) Nuage de points : objets vs longueur moyenne des captions ----
    mean_len = np.array([np.mean([len(c.split()) for c in clean[i]]) for i in imgs])
    plt.figure(figsize=(7, 5))
    plt.scatter(n_obj, mean_len, alpha=0.2, s=10)
    r = float(np.corrcoef(n_obj, mean_len)[0, 1]) if n_obj.std() > 0 and mean_len.std() > 0 else float("nan")
    plt.title(f"Objets par image vs longueur moyenne des captions (r = {r:.2f})")
    plt.xlabel("Nombre d'objets")
    plt.ylabel("Longueur moyenne (mots)")
    save("eda_scatter_objects_vs_caption_length.png")
    stats["corr_objects_vs_caption_length"] = r

    save_json(stats, C.REPORTS / "eda_stats.json")
    print("Stats ->", C.REPORTS / "eda_stats.json")
    print(f"Objets/image: {stats['objects_per_image']['mean']:.1f} | "
          f"Longueur caption: {stats['captions']['length_mean']:.1f} mots | "
          f"Vocabulaire: {stats['captions']['vocab_size']}")


if __name__ == "__main__":
    main()
