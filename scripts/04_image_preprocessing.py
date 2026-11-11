"""ÉTAPE 4 — Prétraitement des images : diagnostic + démonstration (CLIP).

Ce que fait le pipeline (appliqué à la volée à l'étape 6, sans dupliquer les images) :
  1. ouverture + correction de l'orientation EXIF   (common.load_image)
  2. conversion en RGB                              (common.load_image)
  3. redimensionnement : côté court -> 224 px (ratio conservé), puis recadrage central 224x224
  4. normalisation : /255 puis (x - moyenne) / écart-type propres à CLIP
Sorties : results/visualizations/preprocessing_examples.png, results/reports/preprocessing_report.json
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor

import config as C
from common import get_slim, load_image, load_json, save_json


def main():
    kb = load_json(C.SPLITS / "kb_ids.json")
    slim = get_slim("train", C.TRAIN_IMG_DIR)
    paths = {i: C.TRAIN_IMG_DIR / slim["images"][i]["file_name"] for i in kb}

    # --- diagnostic rapide (lit seulement l'en-tête de chaque fichier) ---
    modes, n_exif, sizes = Counter(), 0, []
    for p in tqdm(paths.values(), desc="Diagnostic"):
        with Image.open(p) as im:
            modes[im.mode] += 1
            if im.getexif().get(274, 1) != 1:     # 274 = tag d'orientation EXIF
                n_exif += 1
            sizes.append(im.size)
    print("Modes d'image :", dict(modes), "| orientation EXIF à corriger :", n_exif)

    # --- démonstration avec le processeur CLIP ---
    proc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)
    ip = proc.image_processor
    mean = torch.tensor(ip.image_mean).view(3, 1, 1)
    std = torch.tensor(ip.image_std).view(3, 1, 1)
    sample = random.Random(C.SEED).sample(kb, min(6, len(kb)))
    fig, ax = plt.subplots(2, len(sample), figsize=(2.6 * len(sample), 5.6), squeeze=False)
    for k, i in enumerate(sample):
        img = load_image(paths[i])
        t = ip(images=img, return_tensors="pt")["pixel_values"][0]        # (3, 224, 224)
        ax[0, k].imshow(img)
        ax[0, k].set_title(f"original {img.size[0]}x{img.size[1]}", fontsize=8)
        ax[1, k].imshow((t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax[1, k].set_title("vu par CLIP 224x224", fontsize=8)
        for a in (ax[0, k], ax[1, k]):
            a.axis("off")
    plt.tight_layout()
    plt.savefig(C.VIZ / "preprocessing_examples.png", dpi=150)
    plt.close()

    report = {"image_mode_counts": dict(modes), "n_with_exif_rotation": n_exif,
              "mean_width": sum(s[0] for s in sizes) / len(sizes),
              "mean_height": sum(s[1] for s in sizes) / len(sizes),
              "encoder": C.CLIP_MODEL, "resize": str(ip.size), "crop": str(ip.crop_size),
              "normalize_mean": ip.image_mean, "normalize_std": ip.image_std,
              "tensor_shape": list(t.shape),
              "note": "Redimensionnement à ratio conservé puis recadrage central : les bords "
                      "des images non carrées sont coupés."}
    save_json(report, C.REPORTS / "preprocessing_report.json")
    print("Rapport ->", C.REPORTS / "preprocessing_report.json")


if __name__ == "__main__":
    main()
