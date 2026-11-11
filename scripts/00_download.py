"""ÉTAPE 0 — Télécharge les annotations COCO 2017 + UNIQUEMENT les images nécessaires.
Relançable : ne retélécharge pas ce qui existe déjà.

Usage : python scripts/00_download.py                 (tout)
        python scripts/00_download.py --skip-images   (si le prof a déjà fourni les images)
Si le prof fournit la liste des images de test : mets-la dans data/splits/test_ids_professor.json
(liste d'IDs COCO val2017) AVANT de lancer ce script.
"""
import argparse
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C
from common import load_json, save_json

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
NEEDED = ["captions_train2017.json", "instances_train2017.json",
          "captions_val2017.json", "instances_val2017.json"]


def download(url, dest):
    """Télécharge un fichier (écriture atomique : .part puis renommage)."""
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"  échec {url}: {e}")
        return False


def get_annotations():
    if all((C.ANN_DIR / f).exists() for f in NEEDED):
        print("Annotations déjà présentes.")
        return
    zpath = C.ANN_DIR / "annotations_trainval2017.zip"
    if not zpath.exists():
        print("Téléchargement des annotations (~241 Mo)...")
        with requests.get(ANN_URL, stream=True) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        for name in NEEDED:
            data = z.read(f"annotations/{name}")
            (C.ANN_DIR / name).write_bytes(data)
    print("Annotations extraites dans", C.ANN_DIR)


def get_images(split, n, img_dir, forced_ids=None):
    """Échantillonne n images (seed fixe) ayant >= 5 captions, puis les télécharge."""
    caps = load_json(C.ANN_DIR / f"captions_{split}2017.json")
    imgs = {i["id"]: i["file_name"] for i in caps["images"]}
    if forced_ids:
        chosen = list(forced_ids)
    else:
        count = {}
        for a in caps["annotations"]:
            count[a["image_id"]] = count.get(a["image_id"], 0) + 1
        ok_ids = sorted(i for i in imgs if count.get(i, 0) >= C.N_CAPTIONS)
        random.Random(C.SEED).shuffle(ok_ids)
        chosen = ok_ids[:n]
    jobs = [(f"http://images.cocodataset.org/{split}2017/{imgs[i]}", img_dir / imgs[i])
            for i in chosen]
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(tqdm(ex.map(lambda j: download(*j), jobs), total=len(jobs), desc=split))
    print(f"  {sum(results)}/{len(jobs)} images disponibles pour {split}")
    # manifeste des images ATTENDUES (sert à détecter les images manquantes à l'étape 1)
    save_json({"ids": chosen, "file_names": [imgs[i] for i in chosen]},
              C.RAW / f"{split}_manifest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    get_annotations()
    if not args.skip_images:
        get_images("train", C.N_CANDIDATES, C.TRAIN_IMG_DIR)
        forced = None
        prof_file = C.SPLITS / "test_ids_professor.json"
        if prof_file.exists():
            forced = load_json(prof_file)
            print(f"Liste de test du prof trouvée ({len(forced)} images).")
        get_images("val", C.N_TEST_CANDIDATES, C.VAL_IMG_DIR, forced_ids=forced)
    print("Terminé.")
