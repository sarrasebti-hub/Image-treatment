"""Verify that the knowledge-base index contains only the frozen kb images and no leaking captions.

Checks (all must pass):
  1. indexed image IDs == kb_ids (10,000, no duplicate)
  2. no test image ID and no val image ID in the index
  3. kb_mapping order == kb_image_ids.npy, and every embedding file has one row per image
  4. no indexed (primary) caption is in excluded_captions.json
  5. no indexed caption equals a test or val caption (lowercase, punctuation removed)
  6. index_config.json records 10,000 vectors per index

Output: report/index_id_check.json (exit code 1 if a check fails).
Usage:  PROJECT_DIR=. python src/verify_index.py
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PROJECT_DIR", "."))
norm = lambda s: " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())
load = lambda p: json.load(open(ROOT / p))

kb, val, test = (set(load(f"splits/{s}_ids.json")) for s in ("kb", "val", "test"))
index_ids = [int(x) for x in np.load(ROOT / "embeddings/kb_image_ids.npy")]
mapping = load("embeddings/kb_mapping.json")
excluded = load("splits/excluded_captions.json")
drop = {(e["kb_image_id"], e["caption"]) for e in excluded}
ref = {norm(c) for s in ("val", "test") for it in load(f"splits/{s}_manifest.json") for c in it["captions"]}
shapes = {f: list(np.load(ROOT / f"embeddings/{f}.npy").shape)
          for f in ("kb_clip_image_embeddings", "kb_clip_text_embeddings", "kb_sbert_text_embeddings")}
cfg = load("index/index_config.json")

checks = {
    "indexed_images": len(index_ids),
    "indexed_ids_equal_kb_ids": set(index_ids) == kb,
    "duplicate_indexed_ids": len(index_ids) - len(set(index_ids)),
    "test_ids_in_index": len(set(index_ids) & test),
    "val_ids_in_index": len(set(index_ids) & val),
    "mapping_order_equals_index_ids": [m["image_id"] for m in mapping] == index_ids,
    "embedding_shapes": shapes,
    "all_embeddings_have_one_row_per_image": all(s[0] == len(index_ids) for s in shapes.values()),
    "indexed_captions_in_excluded_list": sum((m["image_id"], m["primary_caption"]) in drop for m in mapping),
    "indexed_captions_equal_to_test_or_val_caption": sum(norm(m["primary_caption"]) in ref for m in mapping),
    "vectors_per_index_in_config": {k: v["total_vectors"] for k, v in cfg["indexes"].items()},
}
passed = (checks["indexed_ids_equal_kb_ids"] and checks["duplicate_indexed_ids"] == 0
          and checks["test_ids_in_index"] == 0 and checks["val_ids_in_index"] == 0
          and checks["mapping_order_equals_index_ids"] and checks["all_embeddings_have_one_row_per_image"]
          and checks["indexed_captions_in_excluded_list"] == 0
          and checks["indexed_captions_equal_to_test_or_val_caption"] == 0
          and all(v == len(index_ids) for v in checks["vectors_per_index_in_config"].values()))
result = {"release": cfg.get("release_version"), "checks": checks, "passed": passed}
(ROOT / "report").mkdir(exist_ok=True)
json.dump(result, open(ROOT / "report/index_id_check.json", "w"), indent=2)
print(json.dumps(result, indent=2))
print("\nAll index checks passed." if passed else "\nINDEX CHECK FAILED.")
sys.exit(0 if passed else 1)
