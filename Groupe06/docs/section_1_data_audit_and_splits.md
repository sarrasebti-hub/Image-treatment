# Data Audit and Frozen Splits

Group 6 (Sparse / BM25 RAG) · Phase 1 · Section written by Member 1 (Data & Splits Lead) · Release v1.1

## 1. Objectives and inputs

This step produces the shared, audited and frozen splits used by every later stage of the project: the RAG knowledge base, hyperparameter tuning and the final evaluation. The inputs are the COCO 2017 images and annotation files (captions_*.json, instances_val2017.json) for train2017 (118,287 images) and val2017 (5,000 images). All steps are implemented in the notebook notebooks/data-audit-and-splits-01.ipynb and in the scripts src/data_audit.py and src/make_splits.py, which give identical results. Every random choice uses the fixed seed 42.

| Split | Source | Images | Captions | Purpose |
|---|---|---|---|---|
| kb (knowledge base) | COCO train2017 | 10,000 | 50,000 (49,777 after caption exclusion, see section 5) | RAG knowledge base |
| val (tuning) | COCO train2017, disjoint from kb | 100 | 500 | Hyperparameter tuning |
| test | COCO val2017, stratified | 1,000 | 5,000 | Final evaluation only |

In total, 10,100 distinct train2017 images are used (kb and val together), not 100,100.

## 2. Data audit

Before any selection, both annotation sets were audited for the following problems: duplicated image IDs, images referenced in the annotations but missing on disk, captions without an image (orphans), empty captions, and images with fewer than five captions.

| Check | train2017 | val2017 |
|---|---|---|
| Images | 118,287 | 5,000 |
| Captions | 591,753 | 25,014 |
| Duplicate image IDs | 0 | 0 |
| Images missing on disk | 0 | 0 |
| Orphan captions | 0 | 0 |
| Empty captions | 0 | 0 |
| Images with fewer than 5 captions | 0 | 0 |
| Images with exactly 5 / 6 / 7 captions | 117,972 / 312 / 3 | 4,987 / 12 / 1 |

No image had to be excluded at this stage. The only irregularity is that a few images carry more than five captions, which is handled by the caption policy in section 4.

## 3. Split creation

**Knowledge base and validation.** All train2017 images were sorted by ID and shuffled once with seed 42. Images were then accepted in shuffled order until 10,100 valid images were collected; the first 10,000 form kb and the last 100 form val. Because both sets come from a single shuffled pool, they are disjoint by construction.

**Image integrity and duplicates.** Each candidate image had to open and decode without error (including conversion to RGB after EXIF orientation correction) and its MD5 content hash had to be new. Unreadable images or images with duplicated content would have been rejected, replaced and logged in split_meta.json. No image was rejected in any split.

**Test set.** The 1,000 test images come from val2017 and are stratified by scene complexity, defined as the number of annotated object instances in instances_val2017.json:

| Stratum | Objects per image | Pool size | Selected |
|---|---|---|---|
| Simple | ≤ 3 | 2,062 | 300 |
| Medium | 4 – 7 | 1,192 | 400 |
| Complex | ≥ 8 | 1,698 | 300 |

The medium stratum receives the largest quota so that typical multi-object scenes are well represented, while simple and complex scenes remain present for error analysis. Each test image keeps its stratum label in test_manifest.json.

**Stored files.** Sorted image IDs (kb_ids.json, val_ids.json, test_ids.json), full manifests with file names and captions, and split_meta.json (seed, quotas, caption policy, rejected samples) are stored in splits/.

## 4. Caption policy

The COCO annotation files provide five reference captions for most images, but a small number of images carry six or seven (312 and 3 images in train2017, 12 and 1 in val2017). To obtain a uniform five captions per image across all splits, the captions of each image are sorted by annotation ID and the first five are kept. Among the selected images, this rule removed 24 captions from 23 knowledge-base images, none from the validation images, and 3 captions from 3 test images, giving exactly 50,000, 500 and 5,000 captions before the exclusion described in section 5. The rule is deterministic, applied identically to every split and independent of caption content, so it introduces no selection bias, and no image was removed because of it. The counts are stored in report/caption_policy_stats.json.

## 5. Anti-leakage control

The leakage audit works at two levels: images and captions.

### 5.1 Image-level checks

1. ID overlap between every pair of splits.
2. Content overlap using MD5 hashes of the image files: train2017 selections (kb and val) against test, and val against kb.
3. Image integrity: every selected image was opened and decoded.

| Check | Result | Expected |
|---|---|---|
| Split sizes (kb / val / test) | 10,000 / 100 / 1,000 | 10,000 / 100 / 1,000 |
| ID overlap kb / val | 0 | 0 |
| ID overlap kb / test | 0 | 0 |
| ID overlap val / test | 0 | 0 |
| Content overlap kb / test | 0 | 0 |
| Content overlap val / kb | 0 | 0 |
| Rejected (unreadable / duplicated content) | 0 / 0 | – |

Because no test image is indexed in the knowledge base, a test image cannot be retrieved for itself. This property is verified explicitly in the retrieval sanity check (Member 3).

### 5.2 Caption-level check

Image-level checks are not enough for a lexical retriever such as BM25. Two different images can have the same caption, because annotators can describe simple scenes with the same short sentence. A test caption that also appears in the knowledge base would then be matched word for word and would inflate retrieval scores artificially. We therefore compared every test and validation caption with every knowledge-base caption at three levels: exact string, lowercase, and normalised (lowercase, punctuation removed, whitespace collapsed; this matches the cleaning used for retrieval).

| Level | Test captions found in kb | Test images affected | Val captions found in kb | Val images affected |
|---|---|---|---|---|
| Exact | 91 | 84 | 8 | 7 |
| Lowercase | 98 | 89 | 9 | 8 |
| Normalised | 101 | 91 | 10 | 9 |

At the normalised level, 91 of the 1,000 test images (9.1%) and 9 of the 100 validation images have at least one caption that also exists in the knowledge base.

### 5.3 Correction applied

To remove this leakage without changing the splits, the matching captions are excluded from the knowledge-base side only. Test and validation captions are left untouched, and no image is removed from any split. Using the normalised level, 223 knowledge-base captions were excluded. They belong to 218 knowledge-base images, with at most two captions removed per image, and no image is left without captions. The knowledge base therefore keeps 10,000 images and 49,777 captions. Each excluded caption is listed in splits/excluded_captions.json together with the test or validation image it matched. The number of excluded captions is larger than the number of matching test and validation captions because one generic caption can occur several times in the knowledge base.

The exclusion applies to the caption index only. kb_manifest.json is unchanged and still lists five captions per image, so every later step (cleaned captions, embeddings, vector index, BM25 index) must read splits/excluded_captions.json and drop those captions before indexing. The counts are stored in report/caption_leakage_stats.json.

## 6. Limitations

- MD5 hashes detect only byte-identical files. Near-duplicates (resized, re-encoded or visually very similar images) are not detected.
- Image content checks cover the selected images (about 11,100), not the full 118,287 train2017 images.
- The caption check detects identical captions after normalisation. Captions that are paraphrases of one another (different wording, same meaning) are not detected.
- The exclusion removes about 0.45% of the knowledge-base captions (223 of 50,000), which is a small and documented loss of retrieval material.
- Test stratification relies only on the number of annotated objects. Other aspects of diversity (interactions, indoor/outdoor context) are not controlled explicitly and are analysed in the data exploration step instead.
- A single random seed was used, so the splits represent one sample of the possible selections.

## 7. Freezing and reproducibility

The ID files (kb_ids.json, val_ids.json, test_ids.json) are frozen since version 1.0 and are unchanged. Version 1.1 keeps exactly the same IDs and adds the caption exclusion list (splits/excluded_captions.json) and the caption-level leakage report. Any further change must be published as a new version.

The notebook and the scripts read their paths from DATA_ROOT and PROJECT_DIR. Both recompute the SHA-256 hashes of the three ID files and compare them with the frozen release. A full re-run of the notebook and of the scripts reproduced the three files exactly (all hashes matched), and gave the same leakage figures, which confirms that the pipeline is reproducible end to end.

**Rules for all groups:** never use test reference captions in retrieval, indexing, prompting or tuning; apply splits/excluded_captions.json before indexing the knowledge base; use val only for hyperparameter tuning and test only for final evaluation.
