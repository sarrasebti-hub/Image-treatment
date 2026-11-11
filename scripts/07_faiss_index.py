"""ÉTAPE 7 — Index FAISS (similarité cosinus) + démonstration de recherche.

Entrées : embeddings/*.npy
Sorties : vector_db/faiss_index.bin (images KB, ids = ids COCO)
          vector_db/faiss_text_index.bin (captions KB, ids = numéro de ligne de text_index.csv)
          vector_db/index_info.json, results/visualizations/faiss_demo.png
Index : IndexFlatIP (recherche exacte, produit scalaire). Les vecteurs sont L2-normalisés,
        donc produit scalaire = cosinus. Pour 10 000 vecteurs, pas besoin d'IVF/HNSW.
"""
import sys
import textwrap
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from common import get_slim, load_image, load_json, save_json

K = 5


def main():
    ids = np.array(load_json(C.EMB_DIR / "image_ids.json"), dtype="int64")
    X = np.load(C.EMB_DIR / "image_embeddings.npy").astype("float32")
    T = np.load(C.EMB_DIR / "text_embeddings.npy").astype("float32")
    assert X.shape[0] == len(ids), "image_embeddings et image_ids ne correspondent pas"
    d = X.shape[1]

    # ---- construction ----
    t0 = time.time()
    index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    index.add_with_ids(X, ids)                                   # on associe chaque vecteur à son id COCO
    build_s = time.time() - t0
    faiss.write_index(index, str(C.VDB_DIR / "faiss_index.bin"))
    tindex = faiss.IndexIDMap(faiss.IndexFlatIP(T.shape[1]))
    tindex.add_with_ids(T, np.arange(len(T), dtype="int64"))
    faiss.write_index(tindex, str(C.VDB_DIR / "faiss_text_index.bin"))

    # ---- démonstration avec des images de validation (jamais dans la KB si split disjoint) ----
    VQ = np.load(C.EMB_DIR / "val_image_embeddings.npy").astype("float32")
    vids = load_json(C.EMB_DIR / "val_image_ids.json")
    t0 = time.time()
    for q in VQ:
        index.search(q[None, :], K)
    latency_ms = 1000 * (time.time() - t0) / len(VQ)

    caps = load_json(C.PROCESSED / "captions_kb.json")
    tr = get_slim("train", C.TRAIN_IMG_DIR)
    qid = vids[0]
    D, I = index.search(VQ[0:1], K + 1)
    hits = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i != qid and i != -1][:K]   # exclut l'image elle-même
    print(f"\nRequête : image {qid}")
    for rank, (i, s) in enumerate(hits, 1):
        print(f"  #{rank} id={i} cosinus={s:.3f} | {caps[str(i)]['raw'][0].strip()}")
    ti = pd.read_csv(C.EMB_DIR / "text_index.csv")
    Dt, It = tindex.search(VQ[0:1], 3)
    print("  Image -> captions les plus proches :")
    for s, r in zip(Dt[0], It[0]):
        row = ti.iloc[int(r)]
        print(f"    {s:.3f} | {caps[str(int(row.image_id))]['raw'][int(row.caption_idx)].strip()}")

    fig, ax = plt.subplots(1, K + 1, figsize=(3 * (K + 1), 4.2))
    ax[0].imshow(load_image(C.TRAIN_IMG_DIR / tr["images"][qid]["file_name"]))
    ax[0].set_title("REQUÊTE", fontsize=9)
    for a, (i, s) in zip(ax[1:], hits):
        a.imshow(load_image(C.TRAIN_IMG_DIR / tr["images"][i]["file_name"]))
        a.set_title(f"cos={s:.2f}\n" + textwrap.fill(caps[str(i)]["raw"][0].strip(), 26), fontsize=7)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(C.VIZ / "faiss_demo.png", dpi=150)
    plt.close()

    info = {"index_type": "IndexIDMap(IndexFlatIP)  (recherche exacte)",
            "metric": "inner product sur vecteurs L2-normalisés = similarité cosinus",
            "dimension": int(d), "n_image_vectors": int(index.ntotal),
            "n_text_vectors": int(tindex.ntotal),
            "build_seconds_images": round(build_s, 4),
            "index_file_mb": round((C.VDB_DIR / "faiss_index.bin").stat().st_size / 1e6, 2),
            "text_index_file_mb": round((C.VDB_DIR / "faiss_text_index.bin").stat().st_size / 1e6, 2),
            "avg_search_latency_ms": round(latency_ms, 3), "faiss_version": faiss.__version__}
    save_json(info, C.VDB_DIR / "index_info.json")
    print("\nIndex FAISS :", info)


if __name__ == "__main__":
    main()
