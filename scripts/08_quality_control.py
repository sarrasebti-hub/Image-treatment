"""ÉTAPE 8 — Contrôle qualité global. Affiche PASS/FAIL pour chaque vérification.
Sortie : results/reports/quality_control.json (code de sortie 1 si un test échoue)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import pandas as pd

import config as C
from common import get_slim, load_json, save_json

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    kb, val, test = (load_json(C.SPLITS / f"{n}_ids.json") for n in ("kb", "val", "test"))
    summ = load_json(C.SPLITS / "splits_summary.json")
    val_in_kb = summ["val_in_kb"]

    # 1) tailles
    check("Taille KB", len(kb) == C.N_KB, f"{len(kb)} / {C.N_KB}")
    check("Taille validation", len(val) == C.N_VAL, f"{len(val)} / {C.N_VAL}")
    check("Taille test", len(test) == C.N_TEST, f"{len(test)} / {C.N_TEST}")

    # 2) ids dupliqués / chevauchements
    check("Pas d'ID dupliqué dans chaque split",
          all(len(s) == len(set(s)) for s in (kb, val, test)))
    check("KB ∩ test = ∅", not (set(kb) & set(test)))
    check("Val ∩ test = ∅", not (set(val) & set(test)))
    check("KB ∩ val = ∅" + (" (validation incluse dans la KB, assumé)" if val_in_kb else ""),
          val_in_kb or not (set(kb) & set(val)))

    # 3) embeddings : forme, NaN, normalisation
    X = np.load(C.EMB_DIR / "image_embeddings.npy")
    T = np.load(C.EMB_DIR / "text_embeddings.npy")
    ids = load_json(C.EMB_DIR / "image_ids.json")
    check("Embeddings images = 1 par image KB", X.shape[0] == len(kb) and ids == kb, X.shape)
    check("Embeddings texte = 5 par image KB", T.shape[0] == len(kb) * C.N_CAPTIONS, T.shape)
    check("Aucun NaN/Inf (images et texte)", np.isfinite(X).all() and np.isfinite(T).all())
    check("Vecteurs normalisés (norme ≈ 1)",
          np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-3)
          and np.allclose(np.linalg.norm(T, axis=1), 1, atol=1e-3))
    check("Aucun vecteur image nul", (np.abs(X).sum(axis=1) > 0).all())
    for n in ("val", "test"):
        E = np.load(C.EMB_DIR / f"{n}_image_embeddings.npy")
        check(f"Embeddings requêtes {n}", E.shape[0] == len(load_json(C.SPLITS / f'{n}_ids.json'))
              and np.isfinite(E).all(), E.shape)

    # 4) associations image-captions (comparaison avec les annotations COCO d'origine)
    caps = load_json(C.PROCESSED / "captions_kb.json")
    tr = get_slim("train", C.TRAIN_IMG_DIR)
    check("5 captions brutes par image KB", all(len(caps[str(i)]["raw"]) == C.N_CAPTIONS for i in kb))
    check("Captions brutes identiques à COCO (non modifiées)",
          all(caps[str(i)]["raw"] == tr["images"][i]["captions"][:C.N_CAPTIONS] for i in kb))
    ti = pd.read_csv(C.EMB_DIR / "text_index.csv")
    exp = [(i, k) for i in kb for k in range(C.N_CAPTIONS)]
    check("text_index.csv aligné avec les embeddings texte",
          len(ti) == len(T) and list(zip(ti.image_id, ti.caption_idx)) == exp)

    # 5) fuite du test : aucune trace des images de test dans les artefacts de la KB
    tset = {str(i) for i in test}
    traces = []
    for f in ("captions_kb.json", "captions_val.json"):
        traces += [(f, k) for k in load_json(C.PROCESSED / f) if k in tset]
    for f in ("coco_objects_kb.json", "coco_objects_val.json",
              "derived_metadata_kb.json", "derived_metadata_val.json"):
        traces += [(f, k) for k in load_json(C.METADATA / f) if k in tset]
    traces += [("text_index.csv", i) for i in set(ti.image_id.astype(str)) if i in tset]
    traces += [("image_ids.json", i) for i in ids if str(i) in tset]
    check("Aucune image de test dans captions / métadonnées / embeddings KB", not traces, traces[:3])
    check("Audit anti-fuite (doublons exacts/visuels test vs KB/val)",
          summ["leakage_audit"]["passed"], summ["leakage_audit"])

    # 6) FAISS
    index = faiss.read_index(str(C.VDB_DIR / "faiss_index.bin"))
    tindex = faiss.read_index(str(C.VDB_DIR / "faiss_text_index.bin"))
    check("FAISS : ntotal = nb images KB", index.ntotal == len(kb), index.ntotal)
    check("FAISS texte : ntotal = nb captions", tindex.ntotal == len(T), tindex.ntotal)
    rng = np.random.default_rng(C.SEED)
    pick = rng.choice(len(kb), size=min(200, len(kb)), replace=False)
    _, I = index.search(X[pick].astype("float32"), 1)
    acc = float(np.mean(I[:, 0] == np.array(kb)[pick]))
    check("FAISS : chaque image KB se retrouve elle-même en top-1", acc >= 0.99, f"{acc:.3f}")
    VQ = np.load(C.EMB_DIR / "val_image_embeddings.npy").astype("float32")
    D, I = index.search(VQ, 5)
    check("FAISS : requêtes validation renvoient 5 résultats valides", (I >= 0).all() and np.isfinite(D).all())
    test_hits = sum(int(i) in set(test) for i in I.flatten())
    check("FAISS : aucun résultat ne pointe vers une image de test", test_hits == 0)

    save_json({"all_passed": all(r["passed"] for r in results), "checks": results},
              C.REPORTS / "quality_control.json")
    n_fail = sum(not r["passed"] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} vérifications réussies.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
