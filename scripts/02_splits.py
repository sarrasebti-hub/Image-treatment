"""ÉTAPE 2 — Création des 3 splits (KB / validation / test) + audit anti-fuite.

Entrées : data/processed/clean_manifest.csv, images val2017 (pour le test)
Sorties : data/splits/{kb,val,test}_ids.json, data/splits/splits_summary.json
Règles  : test = images val2017 (jamais dans train2017) ; aucun doublon exact/visuel
          entre test et KB/validation ; tirage reproductible (SEED).
"""
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import imagehash
import pandas as pd
from PIL import Image

import config as C
from common import (get_slim, load_image, load_json, md5_file, min_distance, save_json)

STRATA = ["simple", "moyenne", "complexe", "tres_complexe"]


def readable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        load_image(path)
        return True
    except Exception:
        return False


def main():
    rng = random.Random(C.SEED)
    manifest = pd.read_csv(C.PROCESSED / "clean_manifest.csv")
    ok = manifest[manifest.status == "ok"].sort_values("image_id")
    ids = ok.image_id.tolist()
    rng.shuffle(ids)
    if len(ids) < C.N_KB:
        sys.exit(f"Seulement {len(ids)} images propres < N_KB={C.N_KB}.")

    # ---------- KB et validation ----------
    kb = sorted(ids[:C.N_KB])
    rest = ids[C.N_KB:]
    if len(rest) >= C.N_VAL:
        val, val_in_kb = sorted(rest[:C.N_VAL]), False
    else:
        val, val_in_kb = sorted(rng.sample(kb, C.N_VAL)), True
        print("ATTENTION : pas assez d'images en plus -> validation tirée DANS la KB.\n"
              "  => à la récupération il faudra exclure l'image elle-même (à confirmer avec le prof).")

    # ---------- Candidats test (val2017) ----------
    slim_val = get_slim("val", C.VAL_IMG_DIR)
    cands = [i for i, v in slim_val["images"].items()
             if len(v["captions"]) >= C.N_CAPTIONS and readable(C.VAL_IMG_DIR / v["file_name"])]
    ref = ok[ok.image_id.isin(set(kb) | set(val))]            # images KB + val
    info = {}
    for i in cands:
        p = C.VAL_IMG_DIR / slim_val["images"][i]["file_name"]
        info[i] = (md5_file(p), str(imagehash.phash(load_image(p))))
    d_min, _ = min_distance([info[i][1] for i in cands], ref.phash.tolist())
    ref_md5 = set(ref.md5)
    removed = [i for i, d in zip(cands, d_min) if info[i][0] in ref_md5 or d <= C.PHASH_THRESHOLD]
    cands = [i for i in cands if i not in set(removed)]

    # ---------- Sélection du test ----------
    prof_file = C.SPLITS / "test_ids_professor.json"
    strata_counts, strata_ranges = {}, {}
    if prof_file.exists():
        test = sorted(load_json(prof_file))
        bad = [i for i in test if i not in set(cands)]
        if bad:
            sys.exit(f"{len(bad)} images de la liste du prof sont introuvables/invalides : {bad[:10]}")
        print("Test = liste fournie par le prof.")
    else:
        df = pd.DataFrame({"image_id": cands,
                           "n_obj": [len(slim_val["images"][i]["objects"]) for i in cands]})
        df["stratum"] = pd.qcut(df["n_obj"].rank(method="first"), 4, labels=STRATA)
        base, extra = divmod(C.N_TEST, 4)
        test = []
        for k, s in enumerate(STRATA):
            sub = df[df.stratum == s]
            pool = sub.image_id.tolist()
            rng.shuffle(pool)
            pick = pool[:base + (1 if k < extra else 0)]
            test += pick
            strata_counts[s] = len(pick)
            strata_ranges[s] = [int(sub.n_obj.min()), int(sub.n_obj.max())]
        if len(test) < C.N_TEST:
            left = [i for i in cands if i not in set(test)]
            rng.shuffle(left)
            test += left[:C.N_TEST - len(test)]
        test = sorted(test)
        if len(test) < C.N_TEST:
            sys.exit(f"Seulement {len(test)} images de test valides < {C.N_TEST}. "
                     "Augmente N_TEST_CANDIDATES puis relance 00_download.py.")

    # ---------- Audit anti-fuite ----------
    audit = {
        "kb_inter_val": len(set(kb) & set(val)),
        "kb_inter_test": len(set(kb) & set(test)),
        "val_inter_test": len(set(val) & set(test)),
        "test_md5_in_kb_or_val": len({info[i][0] for i in test} & ref_md5),
        "test_candidates_removed_for_visual_overlap": len(removed),
    }
    d_test, _ = min_distance([info[i][1] for i in test], ref.phash.tolist())
    audit["test_min_phash_distance_to_kb_val"] = int(d_test.min())
    audit["passed"] = (audit["kb_inter_test"] == 0 and audit["val_inter_test"] == 0
                       and audit["test_md5_in_kb_or_val"] == 0
                       and audit["test_min_phash_distance_to_kb_val"] > C.PHASH_THRESHOLD
                       and (val_in_kb or audit["kb_inter_val"] == 0))
    if not audit["passed"]:
        print("FUITE DETECTEE :", audit)

    # ---------- Sauvegarde (splits figés) ----------
    save_json(kb, C.SPLITS / "kb_ids.json")
    save_json(val, C.SPLITS / "val_ids.json")
    save_json(test, C.SPLITS / "test_ids.json")
    summary = {"seed": C.SEED, "n_kb": len(kb), "n_val": len(val), "n_test": len(test),
               "val_in_kb": val_in_kb, "test_source": "professor_list" if prof_file.exists()
               else "stratified_by_object_count_quartiles(val2017)",
               "test_strata_counts": strata_counts, "test_strata_n_objects_ranges": strata_ranges,
               "leakage_audit": audit,
               "open_questions": ["N_TEST: 100 ou 1000 ?", "validation disjointe de la KB ou incluse ?"]}
    save_json(summary, C.SPLITS / "splits_summary.json")
    print(f"KB={len(kb)}  VAL={len(val)} (dans KB: {val_in_kb})  TEST={len(test)}")
    print("Audit anti-fuite :", "OK" if audit["passed"] else "ECHEC", audit)


if __name__ == "__main__":
    main()
