"""ÉTAPE 6 — Embeddings CLIP (images + captions).

Entrées : splits, captions_kb.json, images
Sorties (dossier embeddings/) :
  image_embeddings.npy      (N_KB, 512)       images de la KB   | ids : image_ids.json
  text_embeddings.npy       (N_KB*5, 512)     captions NETTOYÉES | index : text_index.csv
  val_image_embeddings.npy / test_image_embeddings.npy (+ *_ids.json)  = futures requêtes
  embeddings_info.json      dimension, normalisation, modèle, temps...
Reprise automatique : les morceaux déjà calculés sont dans embeddings/_tmp (supprime ce dossier
si tu changes de modèle ou de splits).
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

import config as C
from common import get_slim, load_image, load_json, save_json

TMP = C.EMB_DIR / "_tmp"
TMP.mkdir(exist_ok=True)

print(f"Chargement de {C.CLIP_MODEL} sur {C.DEVICE}...")
model = CLIPModel.from_pretrained(C.CLIP_MODEL).to(C.DEVICE).eval()
proc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)


def l2(x):
    return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")


def embed_images(paths):
    out = []
    for s in range(0, len(paths), C.BATCH_SIZE):
        imgs = [load_image(p) for p in paths[s:s + C.BATCH_SIZE]]     # EXIF + RGB
        inputs = proc(images=imgs, return_tensors="pt").to(C.DEVICE)  # resize 224 + normalisation
        with torch.no_grad():
            out.append(l2(model.get_image_features(**inputs)))
    return np.concatenate(out)


def embed_texts(texts):
    out = []
    for s in range(0, len(texts), 256):
        tok = proc.tokenizer(texts[s:s + 256], padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(C.DEVICE)
        with torch.no_grad():
            out.append(l2(model.get_text_features(**tok)))
    return np.concatenate(out)


def chunked(items, name, fn, chunk=1024):
    """Calcule par morceaux de 1024 éléments, en sauvegardant chaque morceau (reprise possible)."""
    parts = []
    for k, s in enumerate(tqdm(range(0, len(items), chunk), desc=name)):
        f = TMP / f"{name}_{len(items)}_{k:05d}.npy"
        if not f.exists():
            np.save(f, fn(items[s:s + chunk]))
        parts.append(np.load(f))
    return np.concatenate(parts)


def main():
    kb = load_json(C.SPLITS / "kb_ids.json")
    val = load_json(C.SPLITS / "val_ids.json")
    test = load_json(C.SPLITS / "test_ids.json")
    tr, va = get_slim("train", C.TRAIN_IMG_DIR), get_slim("val", C.VAL_IMG_DIR)
    caps = load_json(C.PROCESSED / "captions_kb.json")
    info = {"model": C.CLIP_MODEL, "device": C.DEVICE, "batch_size": C.BATCH_SIZE,
            "normalization": "L2 (norme = 1) -> produit scalaire = similarité cosinus",
            "dtype": "float32", "text_max_tokens": 77,
            "torch": torch.__version__, "transformers": transformers.__version__}

    # ---- images KB ----
    t0 = time.time()
    kb_paths = [C.TRAIN_IMG_DIR / tr["images"][i]["file_name"] for i in kb]
    X = chunked(kb_paths, "kb_images", embed_images)
    np.save(C.EMB_DIR / "image_embeddings.npy", X)
    save_json(kb, C.EMB_DIR / "image_ids.json")
    info["image_embeddings"] = {"shape": list(X.shape), "seconds": round(time.time() - t0, 1)}

    # ---- captions KB (nettoyées) : 5 par image, ordre = ordre de kb ----
    t0 = time.time()
    rows, texts = [], []
    for i in kb:
        for k, c in enumerate(caps[str(i)]["clean"]):
            rows.append({"row": len(rows), "image_id": i, "caption_idx": k})
            texts.append(c if c else "empty")
    T = chunked(texts, "kb_texts", embed_texts)
    np.save(C.EMB_DIR / "text_embeddings.npy", T)
    pd.DataFrame(rows).to_csv(C.EMB_DIR / "text_index.csv", index=False)
    info["text_embeddings"] = {"shape": list(T.shape), "seconds": round(time.time() - t0, 1)}

    # ---- requêtes futures : validation (train2017) et test (val2017) ----
    for name, ids, slim, folder in [("val", val, tr, C.TRAIN_IMG_DIR), ("test", test, va, C.VAL_IMG_DIR)]:
        paths = [folder / slim["images"][i]["file_name"] for i in ids]
        E = chunked(paths, f"{name}_images", embed_images)
        np.save(C.EMB_DIR / f"{name}_image_embeddings.npy", E)
        save_json(ids, C.EMB_DIR / f"{name}_image_ids.json")
        info[f"{name}_image_embeddings"] = {"shape": list(E.shape)}

    info["embedding_dim"] = int(X.shape[1])
    save_json(info, C.EMB_DIR / "embeddings_info.json")
    print("Terminé.", info["image_embeddings"], info["text_embeddings"])


if __name__ == "__main__":
    main()
