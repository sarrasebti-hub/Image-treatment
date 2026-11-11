"""ÉTAPE 5 — Captions (brutes + nettoyées) et métadonnées.

Sorties :
  data/processed/captions_kb.json, captions_val.json   {id: {"raw": [5], "clean": [5]}}
  data/processed/captions_test_EVAL_ONLY.json          captions brutes du test (évaluation seulement)
  data/metadata/coco_objects_{kb,val}.json             source = annotations COCO (réelles)
  data/metadata/coco_objects_test_EVAL_ONLY.json       idem pour le test (évaluation seulement)
  data/metadata/derived_metadata_{kb,val}.json         source = heuristiques (captions + bbox)
"""
import itertools
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config as C
from common import caption_attributes, clean_caption, get_slim, load_json, save_json, spatial_relation


def objects_entry(img, cat_info):
    objs = [{"category": cat_info[o["category_id"]]["name"],
             "supercategory": cat_info[o["category_id"]]["supercategory"],
             "bbox": o["bbox"], "area": o["area"], "iscrowd": o["iscrowd"]}
            for o in img["objects"]]
    return {"source": "coco_annotations", "n_objects": len(objs),
            "categories": sorted({o["category"] for o in objs}), "objects": objs}


def derived_entry(img, clean_caps, cat_info):
    attrs = caption_attributes(clean_caps)
    objs = sorted([o for o in img["objects"] if not o["iscrowd"] and o["area"] > 0],
                  key=lambda o: -o["area"])[:6]
    rels = [f'{cat_info[a["category_id"]]["name"]} {spatial_relation(a["bbox"], b["bbox"])} '
            f'{cat_info[b["category_id"]]["name"]}' for a, b in itertools.combinations(objs, 2)]
    return {"source": "heuristic_from_captions_and_bboxes", **attrs, "spatial_relations": rels}


def main():
    kb = load_json(C.SPLITS / "kb_ids.json")
    val = load_json(C.SPLITS / "val_ids.json")
    test = load_json(C.SPLITS / "test_ids.json")
    tr = get_slim("train", C.TRAIN_IMG_DIR)
    va = get_slim("val", C.VAL_IMG_DIR)

    for name, ids in [("kb", kb), ("val", val)]:
        caps, objs, derived = {}, {}, {}
        for i in ids:
            img = tr["images"][i]
            raw = img["captions"][:C.N_CAPTIONS]
            clean = [clean_caption(c) for c in raw]
            caps[i] = {"raw": raw, "clean": clean}                 # raw = JAMAIS modifié
            objs[i] = objects_entry(img, tr["categories"])
            derived[i] = derived_entry(img, clean, tr["categories"])
        save_json(caps, C.PROCESSED / f"captions_{name}.json")
        save_json(objs, C.METADATA / f"coco_objects_{name}.json")
        save_json(derived, C.METADATA / f"derived_metadata_{name}.json")
        n_words = [len(c.split()) for v in caps.values() for c in v["clean"]]
        print(f"{name}: {len(ids)} images, {len(n_words)} captions, "
              f"longueur moy. {sum(n_words) / len(n_words):.1f} mots, "
              f"min {min(n_words)}, max {max(n_words)}, "
              f"vocabulaire {len({w for v in caps.values() for c in v['clean'] for w in c.split()})}")

    # test : références d'évaluation SEULEMENT, dans des fichiers séparés
    save_json({i: {"raw": va["images"][i]["captions"][:C.N_CAPTIONS]} for i in test},
              C.PROCESSED / "captions_test_EVAL_ONLY.json")
    save_json({i: objects_entry(va["images"][i], va["categories"]) for i in test},
              C.METADATA / "coco_objects_test_EVAL_ONLY.json")
    print(f"test: {len(test)} images -> fichiers *_EVAL_ONLY (ne jamais utiliser pour la récupération)")


if __name__ == "__main__":
    main()
