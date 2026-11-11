"""ÉTAPE 9 — Assemble le rapport final de la Phase 1 (Markdown) à partir des fichiers produits."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config as C
from common import load_json


def j(path):
    return load_json(path) if Path(path).exists() else {}


def main():
    clean = j(C.REPORTS / "cleaning_report.json")
    sp = j(C.SPLITS / "splits_summary.json")
    eda = j(C.REPORTS / "eda_stats.json")
    pre = j(C.REPORTS / "preprocessing_report.json")
    emb = j(C.EMB_DIR / "embeddings_info.json")
    idx = j(C.VDB_DIR / "index_info.json")
    qc = j(C.REPORTS / "quality_control.json")
    cap, obj = eda.get("captions", {}), eda.get("objects_per_image", {})
    L = ["# Rapport Phase 1 — Exploration et prétraitement des données", "",
         "## 1. Audit et nettoyage",
         f"- Fichiers analysés : {clean.get('n_files_found')}",
         f"- Statuts : {clean.get('status_counts')}",
         f"- Doublons exacts supprimés : {len(clean.get('exact_duplicates', []))}",
         f"- Quasi-doublons visuels signalés : {len(clean.get('visual_duplicates_flagged', []))} "
         f"(supprimés : {clean.get('visual_duplicates_removed')})",
         f"- Images propres : {clean.get('n_clean_images')}", "",
         "## 2. Splits (figés, seed {})".format(sp.get("seed")),
         "| Split | Taille | Source |", "|---|---|---|",
         f"| Base de connaissances | {sp.get('n_kb')} | COCO train2017 |",
         f"| Validation | {sp.get('n_val')} | train2017 (dans la KB : {sp.get('val_in_kb')}) |",
         f"| Test | {sp.get('n_test')} | val2017 — {sp.get('test_source')} |", "",
         f"- Audit anti-fuite : `{sp.get('leakage_audit')}`",
         f"- Strates du test : {sp.get('test_strata_counts')}", "",
         "### Points à confirmer avec le professeur",
         "- Taille du test : 100 ou 1 000 images (le PDF est incohérent) ?",
         "- Validation disjointe de la KB (choix actuel) ou incluse dans la KB ?", "",
         "## 3. Exploration statistique",
         f"- Classes présentes : {eda.get('n_categories_present')} ; top 10 : {eda.get('top10_categories')}",
         f"- Objets/image : moyenne {obj.get('mean', 0):.2f}, médiane {obj.get('median')}, "
         f"max {obj.get('max')}",
         f"- Complexité des scènes : {eda.get('scene_complexity')}",
         f"- Captions : {cap.get('n_captions')} ; longueur moyenne {cap.get('length_mean', 0):.2f} mots "
         f"(min {cap.get('length_min')}, max {cap.get('length_max')}) ; vocabulaire {cap.get('vocab_size')}",
         f"- POS (%) : {eda.get('pos_distribution_pct')}", ""]
    for f in sorted(C.VIZ.glob("eda_*.png")):
        L.append(f"![{f.stem}](../visualizations/{f.name})")
    L += ["", "## 4. Prétraitement des images",
          f"- Modes : {pre.get('image_mode_counts')} ; orientation EXIF corrigée : {pre.get('n_with_exif_rotation')}",
          f"- Encodeur : {pre.get('encoder')} — resize {pre.get('resize')}, crop {pre.get('crop')}",
          f"- Normalisation : mean {pre.get('normalize_mean')}, std {pre.get('normalize_std')}", "",
          "## 5. Captions et métadonnées",
          "- Captions brutes conservées telles quelles ; version nettoyée uniquement pour l'indexation.",
          "- Métadonnées **COCO** (réelles) : `data/metadata/coco_objects_*.json`.",
          "- Métadonnées **dérivées** (heuristiques captions/bbox, pas de modèle IA) : "
          "`data/metadata/derived_metadata_*.json`.",
          "- Fichiers `*_EVAL_ONLY` : références du test, réservées à l'évaluation.", "",
          "## 6. Embeddings",
          f"- Modèle : {emb.get('model')} ; dimension {emb.get('embedding_dim')} ; "
          f"normalisation : {emb.get('normalization')}",
          f"- Images KB : {emb.get('image_embeddings')} ; captions KB : {emb.get('text_embeddings')}", "",
          "## 7. Index FAISS", f"- {idx}", "",
          "![faiss_demo](../visualizations/faiss_demo.png)", "",
          "## 8. Contrôle qualité",
          f"- Tous les tests réussis : **{qc.get('all_passed')}**", "",
          "| Vérification | Résultat |", "|---|---|"]
    for r in qc.get("checks", []):
        L.append(f"| {r['check']} | {'PASS' if r['passed'] else 'FAIL'} |")
    out = C.REPORTS / "phase1_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("Rapport ->", out)


if __name__ == "__main__":
    main()
