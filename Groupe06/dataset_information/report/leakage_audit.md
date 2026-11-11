# Leakage Audit Summary – Splits v1.1

**Scope:** frozen splits `kb` (knowledge base), `val` (validation / tuning) and `test`.

**Raw results:** `report/leakage_audit.json`, `report/audit_report.json` and `report/caption_leakage_stats.json`.

**Code:** `notebooks/data-audit-and-splits-01.ipynb` and `src/make_splits.py`.

## 1. Split definition

| Split | Source | Images | Captions |
|---|---|---|---|
| Knowledge base (`kb`) | COCO train2017 | 10,000 | 50,000 (49,777 after caption exclusion, see section 4) |
| Validation (`val`) | COCO train2017, disjoint from `kb` | 100 | 500 |
| Test (`test`) | COCO val2017, stratified (300 simple / 400 medium / 300 complex) | 1,000 | 5,000 |

Seed: 42. Captions: first 5 per image, ordered by annotation ID.
`kb` and `val` were drawn from one shuffled pool of train2017 (10,100 distinct images), so they are disjoint by construction.

## 2. Checks performed

1. **ID overlap:** set intersections between every pair of splits.
2. **Content overlap:** MD5 hash of every selected image; hashes of train2017 selections (`kb` + `val`) compared with hashes of `test`, and hashes of `val` compared with hashes of `kb`.
3. **Image integrity:** every selected image opened and decoded (RGB, EXIF orientation corrected); unreadable or duplicated-content candidates would have been rejected and replaced.
4. **Caption overlap:** every `test` and `val` caption compared with every `kb` caption at three levels: exact string, lowercase, and normalised (lowercase, punctuation removed, whitespace collapsed).

## 3. Results

### 3.1 Images

| Check | Result | Expected |
|---|---|---|
| `kb` size | 10,000 | 10,000 |
| `val` size | 100 | 100 |
| `test` size | 1,000 | 1,000 |
| ID overlap `kb` / `val` | 0 | 0 |
| ID overlap `kb` / `test` | 0 | 0 |
| ID overlap `val` / `test` | 0 | 0 |
| Content overlap `kb` / `test` | 0 | 0 |
| Content overlap `val` / `kb` | 0 | 0 |
| Rejected: unreadable | 0 | – |
| Rejected: duplicated content | 0 | – |

All assertions passed. No image was removed or replaced.

### 3.2 Captions

| Level | `test` captions found in `kb` | `test` images affected | `val` captions found in `kb` | `val` images affected |
|---|---|---|---|---|
| Exact | 91 | 84 | 8 | 7 |
| Lowercase | 98 | 89 | 9 | 8 |
| Normalised | 101 | 91 | 10 | 9 |

Different images can share the same caption. For a lexical retriever such as BM25, a test caption that also exists in the knowledge base would be matched word for word and would inflate the scores.

## 4. Correction (v1.1)

The matching captions are excluded from the **knowledge-base side only**, using the normalised level. `test` and `val` captions are not touched, and no image is removed from any split.

| Item | Value |
|---|---|
| Excluded `kb` captions | 223 |
| `kb` images affected | 218 |
| Max captions removed from one image | 2 |
| `kb` images left without captions | 0 |
| `kb` captions remaining | 49,777 |

The list is stored in `splits/excluded_captions.json` (each entry gives the `kb` image, the caption, and the `test` / `val` image it matched). `kb_manifest.json` is unchanged: **every later step must drop the listed captions before cleaning, embedding or indexing.**

## 5. Conclusion

No `test` image appears in the knowledge base (by ID or by image content), and the validation set shares no image with the knowledge base. After the exclusion in section 4, no `test` or `val` caption remains in the indexed knowledge base (after normalisation). A test image cannot be retrieved for itself because it is not indexed; this should also be confirmed in the retrieval sanity check.

## 6. Limitations

- MD5 detects byte-identical files only. Near-duplicates (resized, re-encoded or visually very similar images) are not detected.
- Content checks cover the selected images (about 11,100), not the full 118,287 train2017 images.
- The caption check finds identical captions after normalisation. Paraphrases (different wording, same meaning) are not detected.
- The exclusion removes about 0.45% of the knowledge-base captions (223 of 50,000).

## 7. Rules for all groups

- Do not modify the frozen ID files (`kb_ids.json`, `val_ids.json`, `test_ids.json`); any change must be released as a new version.
- Apply `splits/excluded_captions.json` before indexing the knowledge base.
- Never use test reference captions in retrieval, indexing, prompting or tuning.
- Use `val` for hyperparameter tuning and `test` for final evaluation only.
