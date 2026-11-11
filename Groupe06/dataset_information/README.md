# Comparative Study of RAG Techniques for Image Captioning — Group 6 (Sparse / BM25 RAG)

Team submission · **Phase 1: data, preprocessing, metadata, embeddings and knowledge base** · 

Release **v1.1**

This repository contains everything needed to build, audit and reproduce the shared data foundation of the project: frozen and leakage-checked splits of MS COCO 2017, cleaned captions, objects detected directly in the images, embeddings, FAISS indexes, and the exploratory statistics.

---

## 1. Summary

| Item | Value |
|---|---|
| Dataset | MS COCO 2017 (captions and images) |
| Knowledge base (kb) | 10,000 images from `train2017` · 49,777 captions after exclusion |
| Validation (val) | 100 images from `train2017`, disjoint from kb · 500 captions |
| Test | 1,000 images from `val2017`, stratified by scene complexity · 5,000 captions |
| Random seed | 42 |
| Leakage control | image level (ID and MD5) and caption level (223 kb captions excluded) |
| Object metadata | YOLOv8m run on the images, confidence ≥ 0.25 · 78,381 objects · 80 classes |
| Embeddings | Sentence-BERT (384 d), OpenCLIP ViT-B-32 text and image (512 d) |
| Indexes | 3 FAISS `IndexFlatIP` indexes (cosine similarity on L2-normalised vectors) |

Phase 1 status:

| Step | Result | Where |
|---|---|---|
| 1. Data audit and frozen splits | Done | `src/`, `splits/`, `report/`, `docs/section_1_*` |
| 2. Preprocessing and metadata extraction | Done | `notebooks/02-*`, `data/`, `docs/section_2_*` |
| 3. Embeddings, indexes and retrieval sanity check | Done | `embeddings/`, `index/`, `notebooks/03_*`, `docs/section_4_3_*` |
| 4. Exploration (before / after preprocessing) | Done | `exploration/`, `docs/member_4_*` |

---

## 2. Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── captions_processed_v1.1.json   raw, cleaned and tokenised captions (kb, val, test)
│   └── metadata.json                  objects detected in each image (YOLOv8m)
├── splits/                            frozen splits (see splits/README.md)
│   ├── kb_ids.json, val_ids.json, test_ids.json            frozen since v1.0
│   ├── kb_manifest.json, val_manifest.json, test_manifest.json
│   ├── split_meta.json
│   └── excluded_captions.json         223 kb captions to drop before indexing (v1.1)
├── report/                            audit and verification outputs
│   ├── audit_report.json, leakage_audit.json, leakage_audit.md
│   ├── caption_policy_stats.json, caption_leakage_stats.json
│   └── retrieval_sanity_check*.json
├── embeddings/                        .npy vectors and the vector-to-image mapping
├── index/                             FAISS indexes and index_config.json
├── exploration/                       statistics tables, figures and exploration notebook
├── notebooks/
│   ├── data-audit-and-splits-01.ipynb
│   ├── 02-preprocessing-and-metadata.ipynb
│   └── 03_embeddings_and_retrieval_sanity_check.ipynb
├── src/                               scripts: config.py, data_audit.py, make_splits.py
├── docs/                              report sections (one per member) and detector documentation
├── rag_captioning_pipeline.py         pilot RAG prototype (see section 9)
└── evaluate_captions.py               pilot BLEU / ROUGE evaluation (see section 9)
```

The COCO images are **not** stored in this repository. They are read from COCO 2017 with the `file_name` of each manifest.

---

## 3. Data and splits

| Split | Source | Images | Captions | Purpose |
|---|---|---:|---:|---|
| `kb` | COCO train2017 | 10,000 | 49,777 (50,000 before exclusion) | Retrieval knowledge base |
| `val` | COCO train2017, disjoint from kb | 100 | 500 | Hyperparameter tuning |
| `test` | COCO val2017, stratified | 1,000 | 5,000 | Final evaluation only |

- **Selection.** All train2017 images are sorted by ID and shuffled once with seed 42; the first 10,100 valid images give kb (first 10,000) and val (last 100), so they cannot overlap.
- **Image checks.** An image is accepted only if it opens, decodes and converts to RGB (EXIF orientation corrected) and its MD5 content hash is new. In the final splits no image was rejected.
- **Test stratification.** By number of annotated objects: simple (≤ 3 objects) 300, medium (4 to 7) 400, complex (≥ 8) 300.
- **Caption policy.** A few COCO images carry 6 or 7 captions; the captions of each image are sorted by annotation ID and the first five are kept (24 captions dropped from kb, 0 from val, 3 from test).
- **Data audit.** No duplicated image ID, missing image, orphan caption or empty caption in train2017 (118,287 images) or val2017 (5,000 images).

Full details: [`splits/README.md`](splits/README.md) and [`docs/section_1_data_audit_and_splits.md`](docs/section_1_data_audit_and_splits.md).

### 3.1 Leakage control

| Level | Check | Result |
|---|---|---|
| Images | ID overlap between splits | 0 |
| Images | MD5 content overlap (kb/test, val/kb) | 0 |
| Captions | test captions also found in kb (normalised) | 101 captions, 91 test images |
| Captions | val captions also found in kb (normalised) | 10 captions, 9 val images |

Different images can have the same short caption, which would inflate a lexical retriever such as BM25. The matching captions are therefore **excluded from the knowledge-base side only**: 223 kb captions from 218 images are listed in `splits/excluded_captions.json`. No image is removed, no image is left without captions, and test and val captions are untouched. The knowledge base keeps 10,000 images and 49,777 captions. See [`report/leakage_audit.md`](report/leakage_audit.md).

---

## 4. Preprocessing and object metadata

**Caption cleaning** (identical for every split): lowercase; every character that is not a to z, 0 to 9, a space or an apostrophe becomes a space; whitespace is collapsed; the caption is split into tokens. No stopword removal or stemming is applied. Each record in `data/captions_processed_v1.1.json` keeps `captions_raw`, `captions_clean` and `tokens`; the exclusion above is applied to `captions_clean` and `tokens` of the kb.

**Objects from the images.** Objects are detected directly in the 11,100 images with **YOLOv8m** (pretrained COCO weights), confidence threshold **0.25**, and stored in `data/metadata.json` (label, confidence, bounding box). The threshold is 0.25 and not 1 because the objects are predictions of a detector; a threshold of 1 applies only to metadata taken from human-written captions.

| Metadata statistic | Value |
|---|---|
| Images with metadata | 11,100 (kb 10,000 · val 100 · test 1,000) |
| Detected objects | 78,381 (7.06 per image on average) |
| Detected classes | 80 (all COCO classes) |
| Mean / median confidence | 0.646 / 0.682 |
| Images without any detected object | 88 (kb 84 · val 2 · test 2) |

Details: [`docs/section_2_preprocessing_and_metadata.md`](docs/section_2_preprocessing_and_metadata.md) and [`docs/detector_documentation.md`](docs/detector_documentation.md).

---

## 5. Embeddings and knowledge base

| Index | Model | Dimension | Vectors | File |
|---|---|---:|---:|---|
| SBERT caption index | `all-MiniLM-L6-v2` | 384 | 10,000 | `index/kb_sbert_flat_ip.index` |
| CLIP caption index | OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), text encoder | 512 | 10,000 | `index/kb_clip_text_flat_ip.index` |
| CLIP image index | OpenCLIP ViT-B-32, image encoder | 512 | 10,000 | `index/kb_clip_image_flat_ip.index` |

- One caption per kb image is indexed (its *primary caption*). If the first caption of an image is in `excluded_captions.json`, another non-excluded caption of the same image is used (75 captions were replaced).
- `embeddings/kb_mapping.json` and `.csv` link each vector to its image ID, primary caption, object count and detected classes; the order matches `embeddings/kb_image_ids.npy`.
- Checks: the indexed image IDs equal the kb IDs; no test or val image is in the index; no indexed caption equals a test or val caption (normalised).
- Run the checks with PROJECT_DIR=. python src/verify_index.py (result in report/index_id_check.json). 
- Versions, batch size and leakage counts are recorded in `index/index_config.json`.

Details: [`docs/section_4_3_embeddings_and_kb.md`](docs/section_4_3_embeddings_and_kb.md).

---

## 6. Exploration

Statistics are computed before and after preprocessing and saved in `exploration/` (tables in CSV, figures in `exploration/visualizations/`, headline numbers in `exploration_summary.json`): caption length and vocabulary (raw 15,992 and cleaned 9,799 types), the effect of the exclusion, objects per image, most frequent classes, co-occurrence of classes, scene complexity, part of speech, and the relation between number of objects and caption length. See [`docs/member_4_exploration_and_repository.md`](docs/member_4_exploration_and_repository.md).

---

## 7. Installation

```bash
pip install -r requirements.txt
```

PyTorch and FAISS may need an environment-specific installation (CPU or GPU build). A GPU is needed only to re-run the object detection (about 10 minutes on a T4); everything else runs on CPU.

---

## 8. Reproduction

All paths are set with environment variables, so nothing is hard-coded.

```bash
export DATA_ROOT=/path/to/coco2017     # contains train2017/, val2017/, annotations_trainval2017/
export PROJECT_DIR=.                    # splits/ and report/ are written here

python src/data_audit.py                # step 1a: audit -> report/audit_report.json
python src/make_splits.py               # step 1b: splits, leakage audit, caption exclusion
```

`make_splits.py` ends by comparing the SHA-256 of `kb_ids.json`, `val_ids.json` and `test_ids.json` with the frozen release and stops with an error if they differ. The same steps are available interactively in `notebooks/data-audit-and-splits-01.ipynb`.

Next steps (Kaggle or Google Colab notebooks):

| Notebook | Needs | Produces |
|---|---|---|
| `notebooks/02-preprocessing-and-metadata.ipynb` | the three manifests and `excluded_captions.json` (`SPLITS_DIR`), the COCO images (`DATA_ROOT`); a GPU, or `RUN_DETECTION = False` to reuse `metadata.json` | `captions_processed_v1.1.json`, `metadata.json`, `preprocessing_stats.json` |
| `notebooks/03_embeddings_and_retrieval_sanity_check.ipynb` | `splits/` including `excluded_captions.json`, `data/` | `embeddings/`, `index/`, `report/retrieval_sanity_check*.json` |
| exploration notebook in `exploration/` | `data/`, `splits/` | `exploration/` tables and figures |

Caution: file names must match exactly (for example `excluded_captions.json`, without a suffix such as ` (1)`). The notebooks stop with an error if the exclusion file is missing, so the exclusion cannot be skipped silently.

---

## 9. Using the data

```python
import json

kb       = json.load(open("splits/kb_manifest.json"))
excluded = json.load(open("splits/excluded_captions.json"))

# Drop the leaking captions before cleaning / embedding / indexing the knowledge base
drop = {(e["kb_image_id"], e["caption"]) for e in excluded}
for item in kb:
    item["captions"] = [c for c in item["captions"] if (item["image_id"], c) not in drop]
```

Rules for every later step:

1. Use `kb` as the only retrieval corpus, **after** applying `excluded_captions.json`.
2. Use `val` for hyperparameter tuning and `test` for the final evaluation only.
3. Never use test reference captions in retrieval, indexing, prompting or tuning.
4. Never modify the frozen ID files; any change to the splits must be released as a new version.

### Pilot generation and evaluation utilities

`rag_captioning_pipeline.py` is a small prototype (FAISS `IndexFlatIP` retrieval, `google/flan-t5-small` generation, baseline vs. RAG prompts) and `evaluate_captions.py` computes BLEU and ROUGE-L from a JSON file of predictions (`final_rag_vs_baseline_evaluation.json`, not included in this repository). **BM25 retrieval, the group's sparse method, is not implemented in this repository yet** (`rank-bm25` is listed in `requirements.txt`). Results from small pilot runs are not conclusive evidence of the superiority of any method.

---

## 10. Versions

| Release | Content |
|---|---|
| v1.0 | Frozen splits (10,000 / 100 / 1,000), data audit, image-level leakage audit |
| **v1.1** | Same image IDs as v1.0. Adds the caption-level leakage check, `excluded_captions.json` (223 kb captions), `captions_processed_v1.1.json`, and the rebuilt embeddings and indexes |

Any further change to the splits or to the exclusion list must be published as a new release.

---

## 11. Limitations

- **Duplicates.** MD5 detects only byte-identical images; near-duplicates (resized, re-encoded or visually similar) are not detected, and content checks cover the selected images (about 11,100), not all 118,287 train2017 images.
- **Caption overlap.** Only identical captions after normalisation are detected; paraphrases are not.
- **Exclusion cost.** 223 of 50,000 kb captions (0.45%) are removed from the index.
- **Stratification.** Test complexity is defined only by the number of annotated objects; interactions and context are analysed in the exploration step but not controlled in the selection.
- **Object metadata.** The detections are predictions of YOLOv8m with an arbitrary threshold of 0.25, limited to the 80 COCO classes, about a third of them have a confidence below 0.5, and they were not manually verified.
- **Single seed.** The splits are one sample of the possible selections.

---

## 12. Documentation

| File | Content |
|---|---|
| [`splits/README.md`](splits/README.md) | Split files, pipeline and reproduction |
| [`report/leakage_audit.md`](report/leakage_audit.md) | Leakage audit summary (images and captions) |
| [`docs/section_1_data_audit_and_splits.md`](docs/section_1_data_audit_and_splits.md) | Data audit and splits (report section) |
| [`docs/section_2_preprocessing_and_metadata.md`](docs/section_2_preprocessing_and_metadata.md) | Preprocessing and metadata (report section) |
| [`docs/detector_documentation.md`](docs/detector_documentation.md) | Object detector documentation |
| [`docs/section_4_3_embeddings_and_kb.md`](docs/section_4_3_embeddings_and_kb.md) | Embeddings and knowledge base (report section) |
| [`docs/member_4_exploration_and_repository.md`](docs/member_4_exploration_and_repository.md) | Exploration and repository (report section) |

## 13. Team

| Role | Responsibility |
|---|---|
| Member 1 · Data & Splits | Audit, frozen splits, leakage control, scripts in `src/`,final report |
| Member 2 · Preprocessing & Metadata | Caption cleaning, object extraction from images |
| Member 3 · Embeddings & Knowledge Base | Embeddings, FAISS indexes, retrieval sanity check |
| Member 4 · Exploration & Report | Statistics, figures, repository documentation, |

