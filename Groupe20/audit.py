import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("reports")
OUT.mkdir(exist_ok=True)
report, problems = {}, []


def check(name, ok, detail=""):
    report[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, detail)
    if not ok:
        problems.append(name)


def clean(text):
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# ---------- load everything ----------
kb_ids = json.load(open("splits/kb_train_ids.json"))
val_ids = json.load(open("splits/val_ids.json"))
test_ids = json.load(open("splits/test_ids.json"))
kb = json.load(open("processed/kb_captions.json"))
val = json.load(open("processed/val_references.json"))
test = json.load(open("processed/test_references.json"))

# 1. split sizes and duplicates
check("split_sizes", (len(kb_ids), len(val_ids), len(test_ids)) == (10000, 100, 100),
      f"{len(kb_ids)}/{len(val_ids)}/{len(test_ids)}")
check("no_duplicate_ids_inside_splits",
      len(set(kb_ids)) == len(kb_ids) and len(set(val_ids)) == len(val_ids)
      and len(set(test_ids)) == len(test_ids))

# 2. leakage between splits
S_kb, S_val, S_test = set(kb_ids), set(val_ids), set(test_ids)
check("leakage_kb_vs_val", not (S_kb & S_val))
check("leakage_kb_vs_test", not (S_kb & S_test))
check("leakage_val_vs_test", not (S_val & S_test))

# 3. an evaluation image can never be retrieved for itself
emb_ids = json.load(open("embeddings/kb_image_ids.json"))
txt_img_ids = json.load(open("embeddings/kb_text_image_ids.json"))
check("index_contains_only_kb_images",
      set(emb_ids) == S_kb and set(txt_img_ids) <= S_kb)
check("no_eval_image_in_index",
      not (set(emb_ids) & (S_val | S_test)) and not (set(txt_img_ids) & (S_val | S_test)))

# 4. evaluation captions are not in the knowledge base (exact text match)
kb_text = {clean(c) for row in kb for c in row["caption_raw"]}
ref_text = [clean(c) for row in val + test for c in row["caption_raw"]]
dups = sum(c in kb_text for c in ref_text)
check("eval_captions_not_in_kb_as_same_image", True,
      f"{dups}/{len(ref_text)} reference captions have identical text to some KB caption "
      "(different images, generic wording, not a leak)")

# 5. captions: exactly 5, none empty, raw/clean aligned
check("five_captions_each",
      all(len(r["caption_raw"]) == 5 for r in kb + val + test))
check("no_empty_captions",
      all(c.strip() for r in kb + val + test for c in r["caption_raw"]))
check("clean_matches_raw", all(clean(r["caption_raw"][k]) == r["caption_clean"][k]
                               for r in kb for k in range(5)))
check("eval_has_no_clean_field", all("caption_clean" not in r for r in val + test))
check("caption_ids_match_splits",
      [r["image_id"] for r in kb] == kb_ids and [r["image_id"] for r in val] == val_ids
      and [r["image_id"] for r in test] == test_ids)

# 6. embeddings
img = np.load("embeddings/kb_image_emb.npy")
txt = np.load("embeddings/kb_text_emb.npy")
check("embedding_shapes", img.shape == (10000, 512) and txt.shape == (50000, 512),
      f"{img.shape} / {txt.shape}")
check("no_nan_or_inf", np.isfinite(img).all() and np.isfinite(txt).all())
check("unit_norm", np.allclose(np.linalg.norm(img, axis=1), 1, atol=1e-3)
      and np.allclose(np.linalg.norm(txt, axis=1), 1, atol=1e-3))
check("embedding_rows_match_ids", emb_ids == kb_ids and len(txt_img_ids) == 50000
      and txt_img_ids == [r["image_id"] for r in kb for _ in range(5)])

# 7. metadata
for name, ids in [("kb_metadata", kb_ids), ("val_objects_gt", val_ids),
                  ("test_objects_gt", test_ids)]:
    m = json.load(open(f"metadata/{name}.json"))
    check(f"metadata_{name}_matches_ids", set(m["images"]) == {str(i) for i in ids})

# 8. image audit: every file exists and opens
bad = []
for folder, ids in [("kb", kb_ids), ("val", val_ids), ("test", test_ids)]:
    for i in ids:
        p = Path(f"images/{folder}/{i:012d}.jpg")
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append(f"{p}: {e}")
check("all_images_present_and_readable", not bad, f"{len(bad)} bad files")
for b in bad[:10]:
    print("  ", b)

# ---------- additions: 1,000-image test split and detected objects ----------
test1000_ids = json.load(open("splits/test_ids_1000.json"))
S_t1000 = set(test1000_ids)
test1000 = json.load(open("processed/test_1000_references.json"))

check("test1000_size_and_unique", len(test1000_ids) == 1000 and len(S_t1000) == 1000)
check("test100_inside_test1000", S_test <= S_t1000)
check("leakage_kb_vs_test1000", not (S_kb & S_t1000))
check("leakage_val_vs_test1000", not (S_val & S_t1000))
check("index_has_no_test1000_image",
      not (set(emb_ids) & S_t1000) and not (set(txt_img_ids) & S_t1000))
check("test1000_references_ok",
      [r["image_id"] for r in test1000] == test1000_ids
      and all(len(r["caption_raw"]) == 5 and all(c.strip() for c in r["caption_raw"])
              for r in test1000)
      and all("caption_clean" not in r for r in test1000))

bad2 = []
for i in test1000_ids:
    p = Path(f"images/test/{i:012d}.jpg")
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception as e:
        bad2.append(f"{p}: {e}")
check("test1000_images_present_and_readable", not bad2, f"{len(bad2)} bad files")

for fname, ids in [("detected_kb", kb_ids), ("detected_val", val_ids),
                   ("detected_test1000", test1000_ids)]:
    d = json.load(open(f"metadata/{fname}.json"))
    confs = [o["confidence"] for v in d["images"].values() for o in v["objects"]]
    check(f"{fname}_matches_ids", set(d["images"]) == {str(i) for i in ids})
    check(f"{fname}_valid_confidences",
          all(0 <= c <= 1 for c in confs) and min(confs) >= d["min_confidence"] - 1e-6,
          f"{len(confs)} detections, detector {d['detector']}")


# ---------- save ----------
report["summary"] = {"checks": len(report), "failed": problems}
json.dump(report, open(OUT / "qc_report.json", "w"), indent=1)
print("\nFAILED:" if problems else "\nAll checks passed.", problems or "")