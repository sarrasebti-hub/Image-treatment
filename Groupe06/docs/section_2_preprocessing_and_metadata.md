# Preprocessing and Metadata Extraction

Group 6 (Sparse / BM25 RAG) · Phase 1 · Member 2 (Preprocessing & Metadata) · Release v1.1

## 1. Inputs and outputs

The step works on the frozen splits of release v1.1 (see the data audit section): 10,000 knowledge-base images (kb), 100 validation images (val) and 1,000 test images (test), 11,100 images in total. Captions come from `splits/*_manifest.json` and images from COCO train2017 (kb, val) and val2017 (test).

| Output | Content |
|---|---|
| `data/captions_processed_v1.1.json` | Raw, cleaned and tokenised captions for the 11,100 images, with the leaking knowledge-base captions removed (section 3) |
| `data/metadata.json` | Objects detected directly in each image (label, confidence, bounding box) |
| `docs/detector_documentation.md` | Detector, version, threshold and decisions |

The objects are extracted from the images and not from the captions, so every image is tied to its own objects.

## 2. Caption cleaning

Each caption is cleaned with a fixed sequence of rules, applied identically to every split:

1. Convert to lowercase.
2. Replace every character that is not a letter a–z, a digit, a space or an apostrophe by a space. Punctuation is therefore removed, and accented or non-ASCII characters act as word separators.
3. Collapse repeated whitespace and strip the ends.
4. Split on whitespace to obtain the tokens.

Apostrophes are kept, so a word such as "man's" stays one token (696 cleaned captions contain one). No stopword removal, stemming or lemmatisation is applied at this stage. Each record keeps the original text (`captions_raw`), the cleaned text (`captions_clean`) and the token lists (`tokens`), so the raw captions are always available. Because BM25 is built on the cleaned field, these rules define what the retriever sees.

| Statistic (v1.1) | Value |
|---|---|
| Tokens per caption (mean / median / min / max) | 10.48 / 10 / 7 / 49 |
| Knowledge-base vocabulary | 9,799 distinct tokens |
| Knowledge-base tokens | 521,583 |
| Empty cleaned captions | 0 |

## 3. Applying the caption exclusion (v1.1)

The caption-level leakage audit found knowledge-base captions that are identical (after normalisation) to test or validation captions. These 223 captions, listed in `splits/excluded_captions.json`, are removed from the cleaned captions and the tokens of the knowledge base. `captions_raw` and the images are not changed, and test and validation captions are untouched.

| | Captions before | Captions after |
|---|---|---|
| kb (cleaned) | 50,000 | 49,777 |
| val | 500 | 500 |
| test | 5,000 | 5,000 |

The exclusion affects 218 knowledge-base images: 213 keep four cleaned captions and 5 keep three, so every image still has at least three. The other 9,782 images keep five. The v1.1 file is derived from the cleaned output by removing the listed captions; the earlier file `data/captions_processed.json` still contains all 50,000 captions, including the leaking ones, and must not be used.

## 4. Object extraction from images

Objects are detected with **YOLOv8m** (Ultralytics 8.4.207, pretrained COCO weights) on a Tesla T4 GPU. Images are processed one at a time with a confidence threshold of **0.25**.

The threshold is a deliberate choice. A confidence threshold of 1 applies only when the metadata is taken from the human-written COCO captions. Here the objects are predictions of a detector, so a threshold of 1 would remove every object; 0.25 is the usual baseline for YOLO and balances precision and recall.

| Split | Images | Objects | Mean per image | Median | Max | Images without objects |
|---|---|---|---|---|---|---|
| kb | 10,000 | 70,721 | 7.07 | 5 | 51 | 84 |
| val | 100 | 684 | 6.84 | 4.5 | 29 | 2 |
| test | 1,000 | 6,976 | 6.98 | 5 | 77 | 2 |
| **Total** | **11,100** | **78,381** | **7.06** | | | **88** |

- All 80 COCO classes appear at least once. The most frequent are person (24,670), car (3,942), chair (3,352), bottle (2,160) and cup (2,040).
- Confidence: mean 0.646, median 0.682. Of the 78,381 detections, 25,667 (32.7%) have a confidence between 0.25 and 0.5, 19,369 between 0.5 and 0.75, and 33,345 at 0.75 or above.
- 88 images (0.8%) have no detected object. These are kept with an empty object list and are not manually corrected.

## 5. Metadata schema

`data/metadata.json` is one object with the fields `detector`, `ultralytics_version`, `conf_threshold` and `data`. `data` is a list with one entry per image:

```json
{
  "image_id": 9,
  "split": "kb",
  "objects": [
    {"label": "bowl", "conf": 0.955, "bbox": [310.2, 0.0, 630.0, 238.1]},
    {"label": "broccoli", "conf": 0.856, "bbox": [254.2, 231.3, 567.0, 479.0]}
  ]
}
```

`conf` is the detector confidence rounded to three decimals, and `bbox` is `[x_min, y_min, x_max, y_max]` in pixels of the original image.

## 6. Clean images

Resized or normalised copies of the images are not saved. Resizing and normalisation are done on the fly during encoding (by Member 3) with the preprocessing pipeline of the CLIP model, which saves disk space and keeps one single source of truth for the images.

## 7. Quality checks

- The 11,100 entries of `metadata.json` match the image IDs of the three splits exactly (kb, val and test), with no duplicated ID.
- `captions_processed_v1.1.json` has 11,100 entries; every kb image has exactly five raw captions, and the cleaned captions and token lists have the same length for every image.
- No cleaned caption is empty, and the same cleaning rules are used for all splits.
- The number of images without objects (88) was checked after the detection run, together with the mean number of objects per image (7.06).

## 8. Limitations

- The detector only knows the 80 COCO classes; other objects are never reported.
- Detections can be wrong or missed. About a third of them have a confidence below 0.5, and person alone represents 31.5% of all detections, so the metadata is more reliable for frequent classes than for rare ones.
- The threshold of 0.25 is a choice and not a ground truth; a higher value would give fewer but safer objects.
- The detections were not compared with the COCO instance annotations in this step.
- Cleaning removes accented characters and keeps no stopword or stemming information; any such choice is left to the retrieval stage.

## 9. Files and reproduction

| File | Role |
|---|---|
| `notebooks/02-preprocessing-and-metadata.ipynb` | Loads the three manifests, cleans the captions, runs YOLOv8m on the 11,100 images, saves the outputs and checks the counts |
| `data/captions_processed_v1.1.json` | Final captions (use this file) |
| `data/metadata.json` | Detected objects |
| `docs/detector_documentation.md` | Detector documentation |

To reproduce: open the notebook on Kaggle with a GPU, add the COCO 2017 dataset and the splits of release v1.1 as inputs, and run all cells. The notebook produces the cleaned captions and the metadata; the exclusion of section 3 is then applied with `splits/excluded_captions.json`.
