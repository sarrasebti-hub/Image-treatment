# Phase 1: cleaning, splits and statistical visualizations

## 1. Clean dataset

The cleaned release is the `cleaned_dataset` folder. The image files are on Google Drive (https://drive.google.com/drive/folders/1xh4MCusgpVPTY2YH83h9KwDsod2Iqn36); everything else is in this repository. After verification, the original COCO downloads were deleted to free disk space. The selected images, captions and original object annotations are kept in this release.

The cleaning audit covers all 118,287 annotated `train2017` images and all 5,000 annotated `val2017` images: **123,287 images in total**. It found **15 exact duplicates**, **0 missing images** and **0 corrupt images**. After duplicate exclusion, 123,272 valid unique candidates remained. The final 10,100 unique images were sampled from those candidates.

Checks performed:

- Expected image files exist, using the COCO image annotation records as the inventory.
- Pillow file verification and full pixel decoding, with truncated-image loading disabled.
- Image dimensions match the source annotation.
- SHA256 of the file bytes and SHA256 of the decoded RGB pixels plus dimensions detect exact duplicates, even when different file bytes decode to identical pixels.
- One representative is kept per duplicate group, preferring train2017 and then the lowest image ID.
- Selected images have original human captions.
- The delivered image copies are re-opened and fully decoded, and their file and pixel hashes are checked against the manifests.

Evidence: `cleaning_audit.csv` (all source records), `excluded_images.csv` (the 15 rejected duplicates and the original kept for each), `summary.json` and `verification.json`.

Scope limits: these are exact-duplicate checks. Crops, resizes or visually similar images with different decoded pixels are not identified as duplicates. The unused `test2017` and `unlabeled2017` sets were not audited or included. Exclusions apply to this clean release only.

## 2. Documented splits

| Split | Source | Images | Original captions | Membership rule |
|---|---|---:|---:|---|
| Knowledge base | train2017 | 10,000 | 50,036 | Main database |
| Validation | train2017 | 100 | 501 | Subset of the knowledge base |
| Test | val2017 | 100 | 500 | Separate from the entire knowledge base |

There are **10,100 unique images** but 10,200 image files, because the 100 validation images are also copied into their own folder. This overlap follows from the split definition (validation is drawn from within the knowledge base); it is not a duplicate inside the knowledge base.

Each split folder contains:

- `images/`: the selected image files (on Google Drive).
- `image_ids.json`: the fixed COCO image IDs.
- `manifest.json`: image paths, source paths, file and pixel hashes, original captions.
- `captions.json`: filtered COCO-format image and caption annotations.
- `instances.json`: filtered original COCO object annotations and category definitions, kept so the analysis can be repeated without the full COCO downloads.

Selection uses uniform random sampling with a fixed random seed of **42**. The test set is a local random selection, not a common test list shared by all groups and not a stratified selection. These IDs must not change during later development.

Verification requires exactly 10,000/100/100 images, all validation IDs inside the knowledge base, and no overlap between the knowledge base and the test set by image ID, file hash or pixel hash. Caption records and image manifests must agree.

All original captions are preserved. Some COCO images have more than five, which explains the 50,036 and 501 caption totals. No captions were generated or rewritten.

When a validation image is used as a retrieval query, its own image and captions must be excluded from the candidates. Test captions remain evaluation references and must never enter the knowledge base.

Split sizes: following the Phase 1 instructions, validation uses 100 images from within the 10,000 and the test set has exactly 100 separate images. The project description also mentions 100 additional validation images and 1,000 test images; we followed the Phase 1 instructions.

## 3. Statistical visualizations

The `visualizations/` folder contains six matplotlib charts:

| File | What it shows |
|---|---|
| `01_split_sizes.png` | Split memberships and the validation overlap |
| `02_cleaning_results.png` | Source audit results and exclusions |
| `03_caption_length_histograms.png` | Caption word-count distribution for each split |
| `04_objects_per_image_histograms.png` | Annotated object-count distribution for each split |
| `05_object_categories.png` | Top 20 categories by annotated instances in the knowledge base |
| `06_image_dimensions.png` | Original image width and height distributions in the knowledge base |

Supporting data: `per_image_statistics.csv`, `category_counts.csv` (all 80 categories, including zero counts), and `statistics_summary.json`.

Caption lengths are word counts (a Unicode word pattern that keeps internal apostrophes), not encoder token counts. Objects are counted from the existing COCO instance annotations; each annotation counts once, including crowd regions. The object count is a simple scene-complexity proxy, not a full measure of interactions or semantic difficulty. Validation statistics overlap the knowledge-base statistics and must not be added to them. Test statistics are descriptive only and must not be used to select model settings.

These charts describe the selected dataset, except the cleaning chart, which covers the whole audited train2017/val2017 pool. They are based on the COCO annotations; no object detector was run.

The caption analysis (vocabulary, top words, part-of-speech, attributes, indoor/outdoor and day/night context, Zipf plot) is in `notebooks/01_caption_analysis.ipynb`, with its figures in `figures/caption_analysis/`.

## 4. Embeddings

The CLIP and Sentence-BERT embeddings, the FAISS indexes, the quality control and the anti-leakage audit are documented in [`../embeddings/README.md`](../embeddings/README.md).

## Reproduce

Required Python packages: Pillow, numpy, matplotlib. The scripts are in the repository root.

1. Run `python verify_and_visualize.py` to check the dataset and regenerate the charts and their underlying statistics. It uses the subset annotations stored here and does not need the original COCO downloads.
2. To repeat the full-source audit with `clean_and_split.py`, first download the COCO train2017/val2017 images and caption annotations again in the folder layout recorded in that script, then run it into an empty output location. It refuses to overwrite an existing `cleaned_dataset` folder.
