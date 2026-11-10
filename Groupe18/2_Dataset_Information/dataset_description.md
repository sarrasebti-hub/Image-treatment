# Dataset description

## Source and annotation structure

The project uses **MS COCO 2017** `train2017` and `val2017`. COCO image records provide IDs, filenames, and dimensions. The instance annotation source supplies object categories and bounding boxes; the captions annotation source supplies human-written reference captions. The complete original COCO images and annotation dumps are not included in this package.

## Selection and splits

The authoritative `selection_report.json` records seed **20261106** and deterministic rarity-first iterative multilabel greedy selection. It favors category coverage and annotation-derived complexity/grouping proxies; it does not establish representativeness. Exactly-five-caption eligibility was used. The final frozen manifests contain 10,000 train images from `train2017`, 100 validation images from `val2017`, and 1,000 test images from `val2017` (11,100 total). Each selected image has five reference captions (55,500 total). A controlled revision record documents four test-ID replacements using replacement seed 20261107; train and validation manifests were unchanged.

## Image and caption processing

Images are decoded, EXIF-transposed, converted to RGB, resized with aspect ratio preserved using LANCZOS, and centered on a black 224 x 224 JPEG canvas. CLIP mean/std normalization is applied during encoding, not baked into those JPEGs. Historical counts of images whose orientation or color mode changed were not retained.

Caption exports preserve `caption_raw` and `caption_cleaned`. The cleaned field trims whitespace, collapses repeated whitespace, and case-folds text. Training caption embeddings use raw training captions. Test references are not placed in the train-only caption index. Token-frequency analyses use Unicode NFKC normalization and case-folding, retain internal apostrophes, and do not stem or lemmatize; POS token counts use spaCy tokenization and are a separate analysis.

## Metadata and provenance

The included metadata schema and per-split records retain distinct provenance for COCO annotations, pixel-derived attributes, CLIP zero-shot predictions, and box-derived geometric summaries. COCO categories/boxes are annotations; pixel color and edge density are coarse image-level measurements; CLIP context/scene outputs are model predictions with relative, uncalibrated scores; spatial relations are geometric inferences from boxes, not semantic action labels. Edge density is not shape recognition.

## Integrity and leakage controls

Saved QC reports record disjoint split IDs, five captions per image, 11,100 successfully decoded processed images, zero exact duplicate groups, and zero cross-split pHash candidates at Hamming threshold 6 (nine within-split candidates remain screening candidates, not confirmed duplicates). Both saved FAISS indexes are train-only. Held-out image retrieval excludes same-ID results. The test split is held out; test reference captions are not in the training caption index.

## Included dataset files

- Frozen JSON/CSV split manifests, current selection report, and controlled-revision provenance.
- Selected raw/cleaned reference-caption JSON exports.
- Per-split metadata JSON, schema, color/shape summaries, and geometric summaries.
- Existing dataset and caption statistics; train2017-wide statistics describe the source split, while selected-image analyses are limited to the selected sample.

The full COCO image collection, processed image tree, and raw annotation dumps are intentionally excluded. Reproducing image encoding requires obtaining COCO 2017 and restoring the expected project data layout.
