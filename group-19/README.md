# Image captioning on MS-COCO: data preparation (phase 1)

Group 19, assigned technique: **Self-RAG**.
Mini-project "A comparative study of RAG techniques for generating image captions".

This repository contains the phase 1 pipeline (data exploration and preprocessing). It produces the shared artifacts used in the following phases: frozen splits, object metadata, CLIP embeddings and a FAISS knowledge base.

## Repository content

- `01_data_preparation.ipynb`: the full pipeline, 12 documented steps
- `data/`: small outputs (splits, cleaning and audit reports, exploration statistics, documentation of the detector, the embeddings and the index, metadata)
- `figures/`: exploration figures (before and after preprocessing)
- `requirements.txt`

Large files (images, embeddings, FAISS indexes) are not in the repository: see "Large files" below.

## Data

- **Knowledge base (train)**: 10,000 images from COCO train2017 with their 5 captions each (50,000 captions)
- **Validation**: 100 other images from train2017, disjoint from the knowledge base
- **Test**: 1,000 images from val2017 (`test100` is the first 100 of them, kept for comparison)
- Split ids are stored in `data/splits.json` (seed 42). The splits are frozen: the notebook reloads them and never regenerates them.

### Anti-leakage control

`data/audit_report.json` records the checks run by step 12 of the notebook:

- train, validation and test ids are pairwise disjoint
- no image is duplicated across splits (MD5 hash)
- the knowledge base contains train images and train captions only
- no validation or test caption is indexed
- no validation or test image is retrieved for itself

## Pipeline

1. Configuration and COCO annotations
2. Splits and anti-leakage control
3. Image download
4. Data audit (missing, corrupted, grayscale and duplicate images, captions)
5. Object extraction with YOLOv8m (confidence threshold 0.5) and metadata
6. Exploration 1 (before preprocessing)
7. Caption preprocessing: raw captions are kept unchanged for evaluation (`captions`), cleaned captions are used for indexing only (`captions_clean`)
8. Image preprocessing: EXIF orientation, RGB conversion, resize to 224x224
9. Exploration 2 (after preprocessing)
10. CLIP ViT-B/32 embeddings (512 dimensions, L2-normalised) for images and train captions
11. FAISS knowledge base: exact inner-product index (equivalent to cosine similarity on normalised vectors)
12. Leakage audit and quality control

## Key figures

| | Before preprocessing | After preprocessing |
|---|---|---|
| Vocabulary size | 16,852 | 10,225 |
| Average caption length (words) | 10.46 | 10.47 |
| Grayscale images | 24 | 0 |
| Images with EXIF rotation | 0 | 0 |
| Image sizes | variable | 224x224 |

178 images have no detected object. They are kept and flagged (`no_object`) to respect the required split sizes.

Full statistics are in `data/explo1.json` and `data/explo_comparison.csv`; the settings of the detector, the embeddings and the index are in `data/detector_info.json`, `data/embedding_info.json` and `data/index_info.json`.

## Installation

The notebook is designed for **Google Colab with a T4 GPU**. Outside Colab, run `pip install -r requirements.txt` (a CUDA GPU is needed for reasonable speed) and adapt the Google Drive paths in the first cell.

## Usage and reproduction

1. Open `01_data_preparation.ipynb` in Google Colab and select a T4 GPU (Runtime > Change runtime type).
2. Run the cells in order (about 20 to 25 minutes, including the image download). Outputs are written to a Google Drive folder named `captioning` (variable `CAP` in the first cell).
3. To reproduce exactly the same data, copy `data/splits.json` to `captioning/splits/splits.json` before running the notebook. Otherwise the splits are generated with seed 42.

## Large files

Embeddings, FAISS indexes and full metadata are available on Google Drive: [DRIVE_LINK_HERE](https://colab.research.google.com/drive/12IKxFIk2O6iB7Bc1cJFBKBJ9_gpvbZ9i?usp=drive_link)

## Status of the notebook

`01_data_preparation.ipynb` assembles the cells that were run step by step in Google Colab to produce the results of this repository. A complete re-run from scratch in a fresh session has not been validated yet.

## Not included yet

- Attribute (colour), spatial relation and scene context (indoor/outdoor, day/night) metadata
- Part-of-speech distribution, co-occurrence matrix and diversity analysis
- The test subset was drawn at random (seed 42); a stratified common subset, if one is provided, is still to be confirmed.
