# Clean dataset and splits

This folder is the frozen Phase 1 dataset: cleaned COCO images, the three documented splits and descriptive statistics. `DELIVERABLES.md` explains how each Phase 1 deliverable was produced, and `verification.json` holds the results of the final verification run. The embeddings built from this dataset are documented in [`../embeddings/README.md`](../embeddings/README.md).

The image folders (`*/images/`, about 1.7 GB) are too large for GitHub. The complete folder, images included, is on Google Drive:
https://drive.google.com/drive/folders/1xh4MCusgpVPTY2YH83h9KwDsod2Iqn36

## Contents

- `knowledge_base/images/`: 10,000 clean images from COCO train2017.
- `validation/images/`: 100 images drawn from that same knowledge base. They are copies kept in their own folder for convenience, not 100 additional unique images.
- `test/images/`: exactly 100 images from COCO val2017, separate from the knowledge base by image ID, file SHA256 and decoded RGB SHA256.
- Each split has `image_ids.json`, `manifest.json` (paths, hashes and original captions), and `captions.json` in COCO format. Original captions are preserved, including the images that have more than five captions.
- Each split also has `instances.json`: the original COCO object annotations filtered to exactly its image IDs, so the dataset can be used without the full COCO annotation files. Manifest `image_path` values are relative to their split folder; `source` records where each image originally came from.
- `cleaning_audit.csv`: one result for every annotated image in train2017 and val2017.
- `excluded_images.csv`: images excluded from selection (missing, corrupt, duplicate or without captions). Duplicate rows identify the retained original.
- `summary.json`: audit counts and split checks.
- `verification.json`: independent recheck of every image file, split membership, captions and hashes.
- `visualizations/`: six matplotlib PNG charts, per-image statistics, the counts of all 80 COCO categories per split and a statistics summary.

## Cleaning

Every annotated train2017 and val2017 image was read and fully decoded. Missing files, unreadable or truncated images and images whose size did not match their annotation were rejected. Exact duplicate files and images with identical decoded RGB pixels were reduced to one representative before sampling. Similar-looking scenes, crops, resizes and lossy recompressions with different pixels are not caught by this exact-duplicate check. The unused test2017 and unlabeled2017 pools were not audited. After verification, the raw COCO downloads were deleted to free disk space; the selected images, captions, object annotations, audit records and scripts were kept.

## Splits

Images were selected by uniform random sampling with Python random seed 42. No common test ID list was provided for all groups, so the test split was generated locally; it is a random sample, not a stratified one. These IDs stay fixed for the rest of the project.

Following the Phase 1 instructions, the validation set is drawn from within the 10,000 knowledge-base images and the test set contains exactly 100 images. The project description also mentions 100 additional validation images and 1,000 test images; we followed the Phase 1 instructions.

Rules for later phases: when a validation image is used as a query, its own image and reference captions must be excluded from the retrieval candidates. Test captions are evaluation references only and must never enter the knowledge base.

## Reproduce

To check the dataset and regenerate the charts, place the image folders from Google Drive in their split folders and run `python verify_and_visualize.py` from the repository root (requires Pillow, numpy and matplotlib). It uses the subset annotations stored here, does not need the original COCO downloads and does not change split membership.

`clean_and_split.py` is the original cleaning and splitting script. Repeating the full audit requires downloading the COCO train2017/val2017 images and caption annotations again, in the folder layout recorded in the script, and an empty output location: the script refuses to overwrite an existing `cleaned_dataset` folder.
