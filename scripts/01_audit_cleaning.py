"""ÉTAPE 1 — Audit et nettoyage des images de la base de connaissances.

Entrées : data/raw/train2017/*.jpg + annotations COCO
Sorties : data/processed/clean_manifest.csv  (une ligne par image + statut)
          results/reports/cleaning_report.json / .md
Détecte : images manquantes, corrompues, sans annotation, avec < 5 captions,
          doublons exacts (MD5) et quasi-doublons visuels (hash perceptuel).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config as C
from common import (get_slim, load_image, load_json, md5_file, near_duplicate_pairs, save_json)


def audit_one(path, image_id, slim):
    row = {"file_name": path.name, "image_id": image_id, "status": "ok", "problem": "",
           "width": 0, "height": 0, "md5": "", "phash": "", "n_captions": 0, "duplicate_of": -1}
    try:
        with Image.open(path) as im:
            im.verify()                      # détecte les fichiers cassés
        img = load_image(path)               # décodage complet (détecte les fichiers tronqués)
        row["width"], row["height"] = img.size
        row["md5"] = md5_file(path)
        row["phash"] = str(imagehash.phash(img))
    except Exception as e:
        row["status"], row["problem"] = "corrupt", repr(e)[:200]
        return row
    if image_id == -1:
        row["status"], row["problem"] = "no_annotation", "fichier absent des annotations COCO"
        return row
    n = len(slim["images"][image_id]["captions"])
    row["n_captions"] = n
    if n < C.N_CAPTIONS:
        row["status"], row["problem"] = "missing_captions", f"{n} caption(s) seulement"
    return row


def main():
    slim = get_slim("train", C.TRAIN_IMG_DIR)
    name2id = {v["file_name"]: k for k, v in slim["images"].items()}
    files = sorted(C.TRAIN_IMG_DIR.glob("*.jpg"))
    if not files:
        sys.exit("Aucune image dans data/raw/train2017. Lance d'abord 00_download.py")
    print(f"{len(files)} fichiers .jpg trouvés.")

    # 1) audit image par image
    rows = [audit_one(p, name2id.get(p.name, -1), slim) for p in tqdm(files, desc="Audit")]
    df = pd.DataFrame(rows)

    # 2) images attendues mais absentes (si le manifeste de 00_download existe)
    missing = []
    manifest = C.RAW / "train_manifest.json"
    if manifest.exists():
        missing = [f for f in load_json(manifest)["file_names"]
                   if not (C.TRAIN_IMG_DIR / f).exists()]

    # 3) doublons EXACTS (même MD5) : on garde l'image avec le plus petit id
    exact = []
    ok = df[df.status == "ok"].sort_values("image_id")
    for md5, grp in ok.groupby("md5"):
        if len(grp) > 1:
            keep = int(grp.iloc[0].image_id)
            for idx in grp.index[1:]:
                df.loc[idx, "status"] = "duplicate_exact"
                df.loc[idx, "duplicate_of"] = keep
                df.loc[idx, "problem"] = f"copie exacte de {keep}"
                exact.append({"keep": keep, "removed": int(df.loc[idx, "image_id"])})

    # 4) quasi-doublons VISUELS (hash perceptuel)
    ok = df[df.status == "ok"].sort_values("image_id")
    ids, idxs, hashes = ok.image_id.tolist(), ok.index.tolist(), ok.phash.tolist()
    pairs = near_duplicate_pairs(hashes, C.PHASH_THRESHOLD)
    visual = [{"keep": int(ids[i]), "other": int(ids[j]), "distance": d} for i, j, d in pairs]
    if C.REMOVE_VISUAL_DUPLICATES:
        for i, j, d in pairs:
            if df.loc[idxs[j], "status"] == "ok":
                df.loc[idxs[j], "status"] = "duplicate_visual"
                df.loc[idxs[j], "duplicate_of"] = ids[i]

    # 5) sauvegardes
    df.to_csv(C.PROCESSED / "clean_manifest.csv", index=False)
    counts = df.status.value_counts().to_dict()
    n_ok = int(counts.get("ok", 0))
    report = {"n_files_found": len(files), "n_images_missing": len(missing),
              "missing_files": missing[:50], "status_counts": counts,
              "exact_duplicates": exact, "visual_duplicates_flagged": visual,
              "visual_duplicates_removed": C.REMOVE_VISUAL_DUPLICATES,
              "problem_files": df[df.status.isin(["corrupt", "no_annotation", "missing_captions"])]
              [["file_name", "status", "problem"]].to_dict("records"),
              "n_clean_images": n_ok, "phash_threshold": C.PHASH_THRESHOLD}
    save_json(report, C.REPORTS / "cleaning_report.json")
    lines = ["# Rapport de nettoyage", "", f"- Fichiers trouvés : **{len(files)}**",
             f"- Images attendues mais absentes : **{len(missing)}**"]
    lines += [f"- {k} : **{v}**" for k, v in counts.items()]
    lines += [f"- Quasi-doublons visuels signalés : **{len(visual)}** "
              f"({'supprimés' if C.REMOVE_VISUAL_DUPLICATES else 'non supprimés, revue manuelle'})",
              f"- **Images propres disponibles : {n_ok}**"]
    (C.REPORTS / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    need = C.N_KB + C.N_VAL
    if n_ok < need:
        print(f"\nATTENTION : {n_ok} images propres < {need} (KB + validation). "
              "Télécharge plus d'images (augmente N_CANDIDATES) ou confirme le choix de validation.")


if __name__ == "__main__":
    main()
