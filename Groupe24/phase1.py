#!/usr/bin/env python
"""
Phase 1 : exploration et pretraitement des donnees COCO (image captioning + RAG)

Usage (dans le terminal de VS Code, dossier du script) :
    python phase1.py splits    # etape 1 : audit + splits figes + anti-fuite
    python phase1.py stats     # etape 2 : statistiques, graphiques, metadonnees
    python phase1.py embed     # etape 3 : embeddings CLIP / Sentence-BERT
    python phase1.py index     # etape 4 : index FAISS + controle qualite
    python phase1.py all       # tout, dans l'ordre

"""
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np

# ============================ A MODIFIER ============================
# Dossier qui contient (a n'importe quelle profondeur) : captions_train2017.json,
# instances_train2017.json, captions_val2017.json, instances_val2017.json,
# et les dossiers d'images train2017 et val2017.
DATA_DIR = os.environ.get("COCO_DIR", r"C:\Users\SG Info\Downloads\traitement dimage\archive")
OUT_DIR = os.environ.get("OUT_DIR", "phase1_output")
# ====================================================================

SEED = 42
N_KB = int(os.environ.get("N_KB", 10000))
N_VAL = int(os.environ.get("N_VAL", 1000))
N_TEST = int(os.environ.get("N_TEST", 1000))


# ----------------------------- utilitaires -----------------------------
def out(*parts):
    """Chemin dans OUT_DIR (cree les dossiers au besoin)."""
    d = os.path.join(OUT_DIR, *parts[:-1])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, parts[-1])


def find_file(name):
    for dp, _, fn in os.walk(DATA_DIR):
        if name in fn:
            return os.path.join(dp, name)
    raise FileNotFoundError(f"{name} introuvable dans {DATA_DIR}")


def find_img_dir(name):
    """Dossier nomme `name` qui contient directement des .jpg."""
    for dp, _, fn in os.walk(DATA_DIR):
        if os.path.basename(dp) == name and any(f.lower().endswith(".jpg") for f in fn):
            return dp
    raise FileNotFoundError(f"Dossier d'images {name} introuvable dans {DATA_DIR}")


def load_caps(split):
    """Retourne (images {id: info}, legendes {id: [5 legendes]})."""
    with open(find_file(f"captions_{split}.json"), encoding="utf-8") as f:
        d = json.load(f)
    images = {im["id"]: im for im in d["images"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a["caption"].strip())
    return images, by_img


def clean(caption):
    """Version nettoyee, UNIQUEMENT pour l'indexation (jamais pour l'evaluation)."""
    c = re.sub(r"[^a-z0-9' ]+", " ", caption.lower())
    return re.sub(r"\s+", " ", c).strip()


def load_splits():
    ids = []
    for n in ["kb", "val", "test"]:
        with open(out("splits", f"{n}_ids.json"), encoding="utf-8") as f:
            ids.append(json.load(f))
    return ids


def check_img(path):
    """Retourne le hash MD5 du fichier, ou None si l'image est illisible/corrompue."""
    from PIL import Image
    try:
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        with Image.open(path) as im:
            im.verify()
        return h
    except Exception:
        return None


# ------------------------- etape 1 : splits ----------------------------
def stage_splits():
    from tqdm import tqdm

    train_dir, val_dir = find_img_dir("train2017"), find_img_dir("val2017")
    images, by_img = load_caps("train2017")
    vimages, vby = load_caps("val2017")
    print(f"train2017 : {len(images)} images dans le JSON, {sum(map(len, by_img.values()))} legendes")

    # Images valides : fichier present + au moins 5 legendes
    on_disk = set(os.listdir(train_dir))
    valid = sorted(i for i in images
                   if images[i]["file_name"] in on_disk and len(by_img[i]) >= 5)
    print("Images valides (fichier present + 5 legendes) :", len(valid))
    assert len(valid) >= N_KB + N_VAL + 200, "Pas assez d'images valides"

    # Tirage reproductible : KB et validation, disjointes
    random.Random(SEED).shuffle(valid)
    kb = sorted(valid[:N_KB])
    val = sorted(valid[N_KB:N_KB + N_VAL])
    pool = valid[N_KB + N_VAL:]  # images de rechange si une image est mauvaise

    # Test : N_TEST images de val2017, stratifiees par nombre d'objets
    with open(find_file("instances_val2017.json"), encoding="utf-8") as f:
        inst = json.load(f)
    n_obj = Counter(a["image_id"] for a in inst["annotations"])
    del inst
    cands = sorted((i for i in vimages if len(vby[i]) >= 5 and n_obj[i] > 0),
                   key=lambda i: (n_obj[i], i))
    t = len(cands) // 3
    groups = [cands[:t], cands[t:2 * t], cands[2 * t:]]  # simple / moyen / complexe
    sizes = [N_TEST - 2 * (N_TEST // 3), N_TEST // 3, N_TEST // 3]
    rnd = random.Random(SEED)
    test = []
    for g, k in zip(groups, sizes):
        test += rnd.sample(g, k)
    test = sorted(test)
    taken = set(test)
    spare_test = [c for c in cands if c not in taken]
    rnd.shuffle(spare_test)

    # Audit : images corrompues + doublons (hash du contenu)
    seen, problems = {}, []

    def good(i, folder, imgs):
        h = check_img(os.path.join(folder, imgs[i]["file_name"]))
        if h is None:
            problems.append({"image_id": i, "probleme": "corrompue ou illisible"})
            return False
        if h in seen:
            problems.append({"image_id": i, "probleme": f"doublon de l'image {seen[h]}"})
            return False
        seen[h] = i
        return True

    def build(ids, spare, n, folder, imgs, label):
        res = [i for i in tqdm(ids, desc=f"Audit {label}") if good(i, folder, imgs)]
        while len(res) < n:  # remplace les images retirees pour garder la taille exacte
            cand = spare.pop(0)
            if good(cand, folder, imgs):
                res.append(cand)
        return sorted(res)

    val = build(val, pool, N_VAL, train_dir, images, "validation")
    kb = build(kb, pool, N_KB, train_dir, images, "base de connaissances")
    test = build(test, spare_test, N_TEST, val_dir, vimages, "test")

    # Controle anti-fuite
    assert not (set(kb) & set(val)) and not (set(kb) & set(test)) and not (set(val) & set(test))
    print(f"KB : {len(kb)} | Validation : {len(val)} | Test : {len(test)}")
    print(f"Problemes detectes : {len(problems)}")
    for p in problems[:20]:
        print("  ", p)
    print("Anti-fuite : OK (aucune image commune entre les 3 splits)")

    for name, ids in [("kb", kb), ("val", val), ("test", test)]:
        with open(out("splits", f"{name}_ids.json"), "w", encoding="utf-8") as f:
            json.dump(ids, f)
    with open(out("splits", "audit_problemes.json"), "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    with open(out("splits", "rapport_splits.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "kb": len(kb), "val": len(val), "test": len(test),
                   "problemes": len(problems), "fuite": 0,
                   "test_stratifie_par_nb_objets": sizes}, f, indent=2)
    print("Splits et rapport d'audit sauvegardes dans", out("splits", ""))


# ------------------------ etape 2 : statistiques -----------------------
def stage_stats():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kb, val, test = load_splits()
    images, by_img = load_caps("train2017")
    vimages, vby = load_caps("val2017")

    def objects_of(path, wanted):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        names = {c["id"]: c["name"] for c in d["categories"]}
        res = defaultdict(list)
        for a in d["annotations"]:
            if a["image_id"] in wanted:
                res[a["image_id"]].append(names[a["category_id"]])
        return res

    print("Lecture des objets (fichier lourd, ~1-2 min, ~3 Go de RAM)...")
    objs = objects_of(find_file("instances_train2017.json"), set(kb) | set(val))
    objs_test = objects_of(find_file("instances_val2017.json"), set(test))

    # Metadonnees : legendes brutes ET nettoyees, objets
    def meta(ids, imgs, by, ob):
        return [{"image_id": i, "file_name": imgs[i]["file_name"],
                 "captions_raw": by[i],
                 "captions_clean": [clean(c) for c in by[i]],
                 "objects": ob.get(i, [])} for i in ids]

    for name, ids, imgs, by, ob in [("kb", kb, images, by_img, objs),
                                    ("val", val, images, by_img, objs),
                                    ("test_references", test, vimages, vby, objs_test)]:
        with open(out("metadata", f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(meta(ids, imgs, by, ob), f, ensure_ascii=False)

    # 1. Top 50 des classes d'objets
    cnt = Counter(c for i in kb for c in objs[i])
    top50 = cnt.most_common(50)
    print("Classes d'objets :", len(cnt), "| Instances :", sum(cnt.values()))
    print("Top 10 :", top50[:10])
    plt.figure(figsize=(8, 11))
    plt.barh([n for n, _ in top50][::-1], [c for _, c in top50][::-1])
    plt.title("Top 50 des classes d'objets (base de connaissances)")
    plt.xlabel("Nombre d'instances")
    plt.tight_layout()
    plt.savefig(out("stats", "top50_classes.png"), dpi=150)
    plt.close()

    # 2. Complexite des scenes
    n_obj = np.array([len(objs[i]) for i in kb])
    n_cls = np.array([len(set(objs[i])) for i in kb])
    print(f"Objets par image : moyenne {n_obj.mean():.2f} | mediane {np.median(n_obj):.0f} | max {n_obj.max()}")
    print(f"Classes distinctes par image : moyenne {n_cls.mean():.2f}")
    print("Images sans objet annote :", int((n_obj == 0).sum()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(n_obj, bins=range(0, 40), edgecolor="black")
    ax[0].set_title("Objets par image")
    ax[0].set_xlabel("Nombre d'objets")
    ax[1].hist(n_cls, bins=range(0, 20), edgecolor="black")
    ax[1].set_title("Classes distinctes par image")
    ax[1].set_xlabel("Nombre de classes")
    plt.tight_layout()
    plt.savefig(out("stats", "complexite_scenes.png"), dpi=150)
    plt.close()

    # 3. Statistiques textuelles
    tok = lambda s: re.findall(r"[a-z']+", s.lower())
    caps_kb = [c for i in kb for c in by_img[i]]
    lens = np.array([len(tok(c)) for c in caps_kb])
    vocab = Counter(w for c in caps_kb for w in tok(c))
    print("Legendes :", len(caps_kb))
    print(f"Longueur : moyenne {lens.mean():.2f} mots | mediane {np.median(lens):.0f} | "
          f"min {lens.min()} | max {lens.max()}")
    print("Taille du vocabulaire :", len(vocab), "| Mots frequents :", vocab.most_common(10))
    plt.figure(figsize=(6, 4))
    plt.hist(lens, bins=range(0, 40), edgecolor="black")
    plt.title("Longueur des legendes (mots)")
    plt.xlabel("Mots")
    plt.tight_layout()
    plt.savefig(out("stats", "longueur_legendes.png"), dpi=150)
    plt.close()

    # 4. Distribution POS (echantillon de 5000 legendes)
    pos_top = None
    try:
        import nltk
        for r in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "punkt", "punkt_tab"]:
            nltk.download(r, quiet=True)
        sample = random.Random(SEED).sample(caps_kb, min(5000, len(caps_kb)))
        pos = Counter(t for c in sample for _, t in nltk.pos_tag(tok(c)))
        pos_top = pos.most_common(12)
        print("POS (top 12) :", pos_top)
        plt.figure(figsize=(7, 4))
        plt.bar([p for p, _ in pos_top], [c for _, c in pos_top])
        plt.title("Distribution des etiquettes POS (5000 legendes)")
        plt.tight_layout()
        plt.savefig(out("stats", "pos.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("POS ignore :", e, "(il faut une connexion Internet pour telecharger nltk)")

    # 5. Co-occurrence des 20 classes les plus frequentes
    top20 = [n for n, _ in top50[:20]]
    M = np.zeros((len(top20), len(top20)), dtype=int)
    for i in kb:
        s = set(objs[i]) & set(top20)
        for a_, b_ in itertools.product(s, s):
            M[top20.index(a_), top20.index(b_)] += 1
    plt.figure(figsize=(9, 8))
    plt.imshow(np.log1p(M), cmap="viridis")
    plt.xticks(range(len(top20)), top20, rotation=90)
    plt.yticks(range(len(top20)), top20)
    plt.colorbar(label="log(1 + nombre d'images)")
    plt.title("Co-occurrence des 20 classes les plus frequentes")
    plt.tight_layout()
    plt.savefig(out("stats", "cooccurrence.png"), dpi=150)
    plt.close()

    # 6. Diversite : couleurs et contextes (mots-cles dans les legendes)
    colors = ["white", "black", "red", "blue", "green", "yellow", "brown", "orange", "pink", "gray", "purple"]
    indoor = ["kitchen", "room", "bathroom", "bedroom", "table", "indoor", "inside", "office"]
    outdoor = ["street", "field", "beach", "park", "outdoor", "road", "grass", "snow", "mountain", "water"]
    night = ["night", "dark", "evening"]
    low = [c.lower().split() for c in caps_kb]
    pct = lambda words: 100 * sum(any(w in c for w in words) for c in low) / len(low)
    col_pct = {c: round(pct([c]), 1) for c in colors}
    ctx = {"interieur": round(pct(indoor), 1), "exterieur": round(pct(outdoor), 1), "nuit": round(pct(night), 1)}
    print("Couleurs (% de legendes) :", col_pct)
    print("Contexte (% de legendes) :", ctx)
    plt.figure(figsize=(8, 4))
    plt.bar(col_pct.keys(), col_pct.values())
    plt.title("Couleurs citees dans les legendes (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out("stats", "couleurs.png"), dpi=150)
    plt.close()

    with open(out("stats", "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": len(cnt), "objets_par_image_moy": float(n_obj.mean()),
                   "classes_par_image_moy": float(n_cls.mean()), "legendes": len(caps_kb),
                   "longueur_moy": float(lens.mean()), "vocabulaire": len(vocab),
                   "top50": top50, "pos_top12": pos_top, "couleurs": col_pct, "contexte": ctx},
                  f, ensure_ascii=False, indent=2)
    print("Statistiques, figures et metadonnees sauvegardees dans", OUT_DIR)


# ------------------------- etape 3 : embeddings ------------------------
def stage_embed():
    import torch
    from PIL import Image, ImageOps
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Appareil :", device, "(sur CPU, compter environ 30 a 60 min)")
    kb, val, test = load_splits()
    images, by_img = load_caps("train2017")
    vimages, _ = load_caps("val2017")
    train_dir, val_dir = find_img_dir("train2017"), find_img_dir("val2017")

    clip = SentenceTransformer("clip-ViT-B-32", device=device)
    report = {"appareil": device, "modele_image": "clip-ViT-B-32",
              "modele_texte_1": "clip-ViT-B-32 (encodeur texte CLIP)",
              "modele_texte_2": "all-MiniLM-L6-v2 (Sentence-BERT)",
              "normalisation": "L2 (cosinus = produit scalaire)", "batch_size": 64}

    def open_rgb(path):
        im = Image.open(path)
        return ImageOps.exif_transpose(im).convert("RGB")  # EXIF corrige + RGB

    def embed_images(ids, folder, imgs, name):
        path = out("embeddings", f"{name}_img_clip.npy")
        if os.path.exists(path):
            print("Deja fait :", path)
            return
        t0, vecs = time.time(), []
        for s in tqdm(range(0, len(ids), 64), desc=f"Images {name}"):
            batch = [open_rgb(os.path.join(folder, imgs[i]["file_name"])) for i in ids[s:s + 64]]
            vecs.append(clip.encode(batch, batch_size=64, normalize_embeddings=True,
                                    convert_to_numpy=True))
        arr = np.vstack(vecs).astype("float32")
        np.save(path, arr)
        report[f"{name}_img_clip"] = {"forme": list(arr.shape), "secondes": round(time.time() - t0, 1)}

    embed_images(kb, train_dir, images, "kb")
    embed_images(val, train_dir, images, "val")
    embed_images(test, val_dir, vimages, "test")

    # Legendes de la base (nettoyees) : une ligne par legende
    rows = [{"image_id": i, "caption_raw": c, "caption_clean": clean(c)}
            for i in kb for c in by_img[i]]
    with open(out("embeddings", "kb_captions.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    texts = [r["caption_clean"] for r in rows]

    path = out("embeddings", "kb_cap_clip.npy")
    if not os.path.exists(path):
        t0 = time.time()
        arr = clip.encode(texts, batch_size=256, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=True).astype("float32")
        np.save(path, arr)
        report["kb_cap_clip"] = {"forme": list(arr.shape), "secondes": round(time.time() - t0, 1)}

    path = out("embeddings", "kb_cap_sbert.npy")
    if not os.path.exists(path):
        t0 = time.time()
        sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        arr = sbert.encode(texts, batch_size=256, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype("float32")
        np.save(path, arr)
        report["kb_cap_sbert"] = {"forme": list(arr.shape), "secondes": round(time.time() - t0, 1)}

    with open(out("embeddings", "rapport_embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Embeddings sauvegardes dans", out("embeddings", ""))


# --------------------- etape 4 : index FAISS + controle ----------------
def stage_index():
    import faiss

    kb, val, test = load_splits()
    emb = lambda n: np.load(out("embeddings", f"{n}.npy"))
    with open(out("embeddings", "kb_captions.json"), encoding="utf-8") as f:
        rows = json.load(f)
    cap_img_ids = [r["image_id"] for r in rows]

    # Index : plat (exact), produit scalaire sur vecteurs normalises = cosinus
    report, indexes = {}, {}
    for name in ["kb_img_clip", "kb_cap_clip", "kb_cap_sbert"]:
        X = emb(name)
        assert not np.isnan(X).any(), f"NaN dans {name}"
        norms = np.linalg.norm(X, axis=1)
        assert np.allclose(norms, 1, atol=1e-3), f"{name} non normalise"
        t0 = time.time()
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        dt = time.time() - t0
        faiss.write_index(index, out("index", f"{name}.faiss"))
        indexes[name] = index
        report[name] = {"type": "IndexFlatIP (exact)", "metrique": "cosinus (produit scalaire normalise)",
                        "dimension": int(X.shape[1]), "vecteurs": int(X.shape[0]),
                        "taille_Mo": round(os.path.getsize(out("index", f"{name}.faiss")) / 1e6, 1),
                        "temps_indexation_s": round(dt, 3)}
        print(name, report[name])

    # Controle qualite
    assert len(set(kb)) == len(kb) and len(set(val)) == len(val) and len(set(test)) == len(test)
    assert not (set(kb) & set(val)) and not (set(kb) & set(test)) and not (set(val) & set(test))
    assert not (set(cap_img_ids) & (set(val) | set(test))), "Fuite : legendes val/test dans la base"
    assert len(emb("kb_img_clip")) == len(kb) and len(emb("kb_cap_clip")) == len(rows)
    print("Controle : pas de NaN, pas de doublons, pas de fuite, tailles coherentes")

    # Test de bon fonctionnement : recherche pour les images de validation et de test
    kb_set = set(kb)
    for name in ["val", "test"]:
        Q = emb(f"{name}_img_clip")
        _, I = indexes["kb_img_clip"].search(Q, 5)
        assert all(kb[j] in kb_set for j in I.ravel())
        _, I2 = indexes["kb_cap_clip"].search(Q, 5)
        assert all(cap_img_ids[j] in kb_set for j in I2.ravel())
        print(f"Recherche {name} -> KB : OK (image->image et image->legende)")

    # Verification : une image de la KB se retrouve elle-meme en 1er (le pipeline devra l'exclure)
    _, I = indexes["kb_img_clip"].search(emb("kb_img_clip")[:50], 1)
    print("Auto-recuperation KB (doit etre 50/50) :", int((I[:, 0] == np.arange(50)).sum()), "/ 50")

    with open(out("index", "rapport_index.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Phase 1 terminee. Rapport :", out("index", "rapport_index.json"))


# --------------------------------- main --------------------------------
STAGES = {"splits": stage_splits, "stats": stage_stats, "embed": stage_embed, "index": stage_index}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1 COCO captioning RAG")
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    for s in (STAGES if args.stage == "all" else [args.stage]):
        print(f"\n===== Etape : {s} =====")
        STAGES[s]()
