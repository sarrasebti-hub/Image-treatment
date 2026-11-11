# Data Audit & Frozen Splits (v1.1)

Phase 1, Part 1 of the project **A comparative study of RAG techniques for image captioning** (Group 6 – Sparse / BM25 RAG).

This part provides the **shared, audited and frozen splits** used by every later step: the RAG knowledge base, hyperparameter tuning and final evaluation.

---

## 1. Summary

| Split | Source | Images | Captions | Purpose |
|---|---|---|---|---|
| `kb` | COCO train2017 | 10,000 | 50,000 (49,777 after exclusion) | RAG knowledge base |
| `val` | COCO train2017 (disjoint from `kb`) | 100 | 500 | Hyperparameter tuning |
| `test` | COCO val2017 (stratified) | 1,000 | 5,000 | Final evaluation only |

- Random seed: **42**
- Captions: first **5** per image, ordered by annotation ID
- Total train2017 images used: 10,100 (`kb` and `val` are drawn from one shuffled pool, so they cannot overlap)
- Test stratification by scene complexity (number of annotated objects):

| Stratum | Objects / image | Pool | Selected |
|---|---|---|---|
| Simple | ≤ 3 | 2,062 | 300 |
| Medium | 4 – 7 | 1,192 | 400 |
| Complex | ≥ 8 | 1,698 | 300 |

---

## 2. Pipeline

```
COCO annotations + images
        │
        ▼
 [1] Data audit  ──────────►  report/audit_report.json
        │
        ▼
 [2] Candidate pools (≥ 5 captions per image)
        │
        ▼
 [3] Seeded sampling  +  image check (decode, EXIF → RGB, MD5)
        │              unreadable / duplicate content → rejected and replaced
        ▼
 [4] Split creation  ─────►  splits/*.json
        │
        ▼
 [5] Leakage audit (images: ID + MD5, captions: exact / lowercase / normalised)
        │          ─────►  report/leakage_audit.json, caption_leakage_stats.json, leakage_audit.md
        ▼
 [6] Caption exclusion (kb side only)  ─►  splits/excluded_captions.json
        │
        ▼
 [7] Freeze  ─────────────►  release v1.1 (same IDs as v1.0)
```

### 2.1 Data audit
Checked on the caption files of train2017 and val2017:
number of images and captions, duplicated image IDs, images missing on disk, orphan captions, empty captions, captions per image.

| Check | train2017 | val2017 |
|---|---|---|
| Images | 118,287 | 5,000 |
| Captions | 591,753 | 25,014 |
| Duplicated image IDs | 0 | 0 |
| Images missing on disk | 0 | 0 |
| Orphan / empty captions | 0 / 0 | 0 / 0 |
| Images with < 5 captions | 0 | 0 |

A few images have 6 or 7 captions (train2017: 312 with 6 and 3 with 7; val2017: 12 with 6 and 1 with 7). Only the first 5 are kept so every image has the same structure.

### 2.2 Image integrity
Every selected image is opened, fully decoded, converted to RGB after EXIF orientation correction, and hashed (MD5). Unreadable files or duplicated contents are rejected and replaced by the next candidate.
**Result: 0 unreadable, 0 duplicated, no replacement needed.**

### 2.3 Leakage audit

| Check | Result |
|---|---|
| ID overlap `kb` / `val`, `kb` / `test`, `val` / `test` | 0 / 0 / 0 |
| Content overlap `kb` / `test` | 0 |
| Content overlap `val` / `kb` | 0 |
| `test` captions also found in `kb` (normalised) | 101 (91 images) |
| `val` captions also found in `kb` (normalised) | 10 (9 images) |

Different images can share the same caption, which would inflate lexical retrieval (BM25). These captions are therefore **excluded from the knowledge-base side**: 223 `kb` captions from 218 images are listed in `excluded_captions.json`, leaving 49,777 captions. No image is removed and no image is left without captions.
Details: [`../report/leakage_audit.md`](../report/leakage_audit.md).

---

## 3. Files

```
splits/
├── kb_ids.json              sorted image IDs of the knowledge base        (frozen v1.0)
├── val_ids.json             sorted image IDs of the validation set        (frozen v1.0)
├── test_ids.json            sorted image IDs of the test set              (frozen v1.0)
├── kb_manifest.json         image_id, file_name, 5 captions
├── val_manifest.json        image_id, file_name, 5 captions
├── test_manifest.json       image_id, file_name, 5 captions, stratum
├── split_meta.json          seed, quotas, caption policy, rejected images
└── excluded_captions.json   kb captions to drop before indexing           (new in v1.1)
report/
├── audit_report.json        data audit + selected-images check
├── caption_policy_stats.json  captions dropped by the "first 5" rule
├── leakage_audit.json       raw image-level leakage results
├── caption_leakage_stats.json caption-level leakage + exclusion counts
└── leakage_audit.md         leakage audit summary
notebooks/
└── data-audit-and-splits-01.ipynb
src/
├── config.py
├── data_audit.py
└── make_splits.py
```

Manifest entry example:

```json
{
  "image_id": 139,
  "file_name": "000000000139.jpg",
  "captions": ["...", "...", "...", "...", "..."],
  "stratum": "medium"
}
```
(`stratum` is present in `test_manifest.json` only.)

`excluded_captions.json` entry:

```json
{
  "kb_image_id": 123,
  "caption": "...",
  "matched_with": [{"split": "test", "image_id": 456}]
}
```

---

## 4. Usage

```python
import json

kb   = json.load(open("splits/kb_manifest.json"))
val  = json.load(open("splits/val_manifest.json"))
test = json.load(open("splits/test_manifest.json"))
excluded = json.load(open("splits/excluded_captions.json"))

print(len(kb), len(val), len(test))   # 10000 100 1000

# Drop leaking captions from the knowledge base before cleaning / indexing
drop = {(e["kb_image_id"], e["caption"]) for e in excluded}
for item in kb:
    item["captions"] = [c for c in item["captions"] if (item["image_id"], c) not in drop]
```

Image files are **not** stored in this repository. Load them from COCO 2017
(`train2017/` for `kb` and `val`, `val2017/` for `test`) using `file_name`.

---

## 5. Reproduction

**Notebook (Kaggle):**
1. Open `notebooks/data-audit-and-splits-01.ipynb` on Kaggle.
2. Add the COCO 2017 dataset as input (used here: `sabahesaraki/2017-2017`).
3. Run all cells (about 15 minutes). The fixed seed regenerates exactly the same splits.
4. Section 8 prints `MATCH` for the three ID files when they are identical to the frozen release.

**Scripts:**
```bash
export DATA_ROOT=/path/to/coco2017      # contains train2017/, val2017/, annotations_trainval2017/
export PROJECT_DIR=/path/to/output      # splits/ and report/ are written here
python src/data_audit.py
python src/make_splits.py
```
`make_splits.py` ends with `MATCH` for `kb_ids`, `val_ids` and `test_ids` (exit code 1 otherwise).

---

## 6. Rules for all groups

- Use `kb` as the only retrieval corpus, **after applying `excluded_captions.json`**.
- Use `val` for hyperparameter tuning and `test` for final evaluation only.
- Never use test reference captions in retrieval, indexing, prompting or tuning.
- Never modify the frozen ID files.

---

## 7. Limitations

- Duplicate detection uses MD5 and finds **byte-identical** files only. Near-duplicates (resized, re-encoded or visually very similar images) are not detected.
- Content checks cover the selected images (about 11,100), not all 118,287 train2017 images.
- The caption check finds identical captions after normalisation; paraphrases are not detected.
- Complexity strata are based on the number of annotated objects only; they do not capture interactions or context.

---

## 8. Version

| Version | Description |
|---|---|
| v1.0 | Frozen splits: 10,000 / 100 / 1,000, audit and image-level leakage reports |
| v1.1 | Same IDs as v1.0. Adds caption-level leakage check and `excluded_captions.json` (223 `kb` captions) |
