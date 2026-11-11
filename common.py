"""Fonctions utilitaires partagées par tous les scripts."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config as C


# ============================ JSON ============================
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================ Images ============================
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path):
    """Ouvre une image, corrige l'orientation EXIF et convertit en RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# ============ Annotations COCO (version allégée, mise en cache) ============
def _fix_keys(slim):
    slim["images"] = {int(k): v for k, v in slim["images"].items()}
    slim["categories"] = {int(k): v for k, v in slim["categories"].items()}
    return slim


def get_slim(split, img_dir):
    """Retourne les annotations COCO UNIQUEMENT pour les images présentes dans img_dir.

    Les gros fichiers instances_*.json (448 Mo pour train) ne sont lus qu'une seule fois ;
    ensuite on réutilise data/processed/slim_<split>.json (quelques Mo).
    Structure : {"categories": {id: {...}}, "images": {id: {file_name, width, height,
                 captions: [5 captions BRUTES], objects: [{category_id,bbox,area,iscrowd}]}}}
    """
    cache = C.PROCESSED / f"slim_{split}.json"
    files = sorted(p.name for p in Path(img_dir).glob("*.jpg"))
    if cache.exists():
        slim = load_json(cache)
        if slim["n_files"] == len(files):
            return _fix_keys(slim)
    print(f"[common] Construction du cache d'annotations '{split}' "
          "(une seule fois, peut utiliser 2-4 Go de RAM)...")
    caps = load_json(C.ANN_DIR / f"captions_{split}2017.json")
    inst = load_json(C.ANN_DIR / f"instances_{split}2017.json")
    fileset = set(files)
    images = {}
    for im in caps["images"]:
        if im["file_name"] in fileset:
            images[im["id"]] = {"file_name": im["file_name"], "width": im["width"],
                                "height": im["height"], "captions": [], "objects": []}
    # captions triées par id d'annotation, texte BRUT conservé tel quel
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        if a["image_id"] in images:
            images[a["image_id"]]["captions"].append(a["caption"])
    for a in inst["annotations"]:
        if a["image_id"] in images:
            images[a["image_id"]]["objects"].append(
                {"category_id": a["category_id"], "bbox": a["bbox"],
                 "area": a["area"], "iscrowd": a["iscrowd"]})
    known = {v["file_name"] for v in images.values()}
    slim = {"split": split, "n_files": len(files),
            "categories": {str(c["id"]): {"name": c["name"], "supercategory": c["supercategory"]}
                           for c in inst["categories"]},
            "images": {str(k): v for k, v in images.items()},
            "unmatched_files": sorted(fileset - known)}
    del caps, inst
    save_json(slim, cache)
    return _fix_keys(slim)


# ============================ Légendes ============================
def clean_caption(text):
    """Nettoyage pour l'INDEXATION uniquement (jamais pour l'évaluation)."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ======== Attributs / contexte : heuristiques à base de mots-clés ========
COLORS = {"white", "black", "red", "blue", "green", "yellow", "brown", "orange",
          "pink", "gray", "grey", "purple"}
SHAPES = {"round", "square", "rectangular", "circular", "oval", "triangular", "long", "tall"}
INDOOR = {"kitchen", "bathroom", "bedroom", "living", "room", "indoor", "indoors", "inside",
          "office", "restaurant", "toilet", "sink", "couch", "sofa", "bed", "desk"}
OUTDOOR = {"street", "road", "field", "beach", "park", "outside", "outdoor", "outdoors",
           "grass", "snow", "sky", "mountain", "water", "ocean", "forest", "city",
           "sidewalk", "sand", "lake", "river", "yard"}
NIGHT = {"night", "dark", "nighttime"}
DAY = {"day", "daytime", "sunny", "sunlight", "morning", "afternoon", "bright"}


def caption_attributes(clean_captions):
    """Déduit couleurs / formes / intérieur-extérieur / jour-nuit des captions (heuristique)."""
    words = {w for c in clean_captions for w in c.split()}
    colors = sorted({"gray" if w == "grey" else w for w in words & COLORS})
    ind, out = bool(words & INDOOR), bool(words & OUTDOOR)
    setting = "ambiguous" if (ind and out) else "indoor" if ind else "outdoor" if out else "unknown"
    nig, day = bool(words & NIGHT), bool(words & DAY)
    time = "ambiguous" if (nig and day) else "night" if nig else "day" if day else "unknown"
    return {"colors": colors, "shapes": sorted(words & SHAPES), "setting": setting, "time": time}


# ======================= Relations spatiales (bbox) =======================
def spatial_relation(b1, b2):
    """Relation de l'objet 1 par rapport à l'objet 2. bbox COCO = [x, y, w, h] (y vers le bas)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    a1, a2 = w1 * h1, w2 * h2
    if inter > 0:
        if inter >= 0.9 * min(a1, a2):
            return "inside" if a1 < a2 else "contains"
        return "overlapping"
    dx, dy = (x1 + w1 / 2) - (x2 + w2 / 2), (y1 + h1 / 2) - (y2 + h2 / 2)
    if abs(dx) >= abs(dy):
        return "left of" if dx < 0 else "right of"
    return "above" if dy < 0 else "below"


# ========================= Hash perceptuel (quasi-doublons) =========================
def _bits(hexes):
    return np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
                     for h in hexes], dtype=np.float32)


def _hamming(a, b):
    return a @ (1 - b).T + (1 - a) @ b.T


def near_duplicate_pairs(hexes, threshold, chunk=1000):
    """Paires (i, j, distance) avec i < j dont la distance de Hamming <= threshold."""
    B = _bits(hexes)
    pairs = []
    for s in range(0, len(B), chunk):
        d = _hamming(B[s:s + chunk], B)
        ii, jj = np.where(d <= threshold)
        for i, j in zip(ii, jj):
            if s + i < j:
                pairs.append((int(s + i), int(j), int(d[i, j])))
    return pairs


def min_distance(hexes_a, hexes_b):
    """Pour chaque hash de A : distance minimale vers B et indice du plus proche."""
    d = _hamming(_bits(hexes_a), _bits(hexes_b))
    return d.min(axis=1), d.argmin(axis=1)
