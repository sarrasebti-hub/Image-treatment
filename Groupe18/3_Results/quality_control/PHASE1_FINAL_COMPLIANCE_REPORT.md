# Phase 1 Final Compliance Audit

## Audit scope and assignment requirements

This audit uses the original assignment PDF, project source, frozen manifests, generated artifacts, and `reports/final_quality_control.json`. The detailed dataset and preprocessing requirements specify 1,000 test images. One occurrence contains the typographical value `1,00`; the detailed requirements and data-audit instructions establish 1,000 as intended. The existing 1,000-image test set matches that requirement and remains unchanged.

## Project and dataset

This Phase 1 project explores COCO, selects fixed splits, preprocesses images, extracts annotation/caption metadata, computes CLIP image and text vectors, builds train-only FAISS indexes, and runs retrieval, POS, perceptual-hash, visual-analysis, and quality-control stages. It does not train a caption-generation model or evaluate generated captions.

| Split | Images | Captions | Processed image checks |
|---|---:|---:|---|
| Train | 10,000 | 50,000 | QC: 10,000 decoded, no missing files |
| Validation | 100 | 500 | QC: 100 decoded, no missing files |
| Test | 1,000 | 5,000 | QC: 1,000 decoded, no missing files |
| Total | 11,100 | 55,500 | 11,100 |

The split manifests are frozen and the controlled replacement integrity checks pass. QC reports no split overlap, no exact SHA-256 duplicates, zero cross-split pHash candidates, and no unresolved current cross-split pairs. The current pHash report has nine within-split review candidates; these are not automatically duplicates. The historical replacement evidence and backups are retained.

## Image preprocessing and integrity

`src/pipeline.py` applies `ImageOps.exif_transpose`, converts images to RGB, resizes with aspect ratio preserved using Pillow thumbnail/LANCZOS, and centers the result on a black 224×224 canvas. The raw files are read only. Model normalization is applied by the CLIP processor during embedding extraction; configured CLIP mean/std are not baked into saved JPEGs. Quality-control and preprocessing reports verify output counts, decodability, and dimensions, and image embedding ID mappings match manifest order.

The number of source images whose EXIF orientation changed and the number requiring grayscale/RGBA conversion were not recorded during preprocessing. These historical counts cannot be reconstructed from the processed JPEGs. Thus the preprocessing behavior is verified from code, but those requested per-image counts and an independent raw-to-processed pixel orientation comparison are not verifiable from retained evidence.

## Captions and caption statistics

Selected caption records contain `caption_raw`, `caption_cleaned`, `caption_id`, and `image_id`. The selection implementation preserves the COCO caption text as `caption_raw`; `caption_cleaned` applies whitespace collapse/trim and case-folding only. Text embeddings explicitly use the raw training captions. QC confirms 55,500 records in split-specific outputs and exactly five captions per selected image. Source-by-source exact caption-ID/text mapping was not part of the existing QC checks; evaluation of generated captions is not implemented in this phase, so no BLEU/METEOR/ROUGE-L/CIDEr/SPICE/BERTScore results are claimed.

Statistical outputs include caption length and top-word frequency tables and length/frequency visualizations. POS processing covers all 55,500 captions (642,147 tokens, zero malformed records in QC). The new `src/extended_analysis.py` report tokenizes raw captions in all frozen splits using Unicode NFKC, case folding, Unicode letter/number tokens, and retained internal apostrophes (without stemming or lemmatization). It reports 591,664 tokens and 9,793 unique tokens across 55,500 captions, per-split statistics, caption lengths, top-500 JSON frequencies, full frequency CSV, and token-count CSV. Distinct-word counts depend on the stated tokenizer.

## Objects, relationships, and visual diversity

COCO instance annotations provide category labels, bounding boxes, area, crowd flags, and object counts in `data/metadata/{split}.json`. Statistical analysis reports category frequencies, object-count distributions, dimensions, and category co-occurrence. The source is COCO ground-truth instance annotation, separate from CLIP predictions.

COCO instances do not provide semantic relation labels. The new analysis derives geometric relations from boxes in 11,100 selected images: centroid ordering for left/right and above/below, strict containment, IoU >= 0.10 for overlap, and normalized center distance <= 0.10 for near. Pair-level CSV, per-image JSON, summary JSON and a frequency chart are saved. These are geometric inferences, not semantic annotations; they do not establish actions, contact, support, or function.

Visual analysis covers CLIP zero-shot scene/context/day-night prompts with confidence scores and abstention, plus pixel-level color, luminance, saturation, and edge-density analysis across 11,100 images. Coarse HSV family shares, dominant family, luminance and saturation moments, and global edge-density proxy are reported in JSON/CSV and visualizations. No independent, validated shape-recognition analysis is available. The edge proxy is not shape recognition, object segmentation, or semantic shape classification; colors are pixel-level and letterbox pixels affect non-square images.

## Embeddings and vector indexes

Image and text embeddings use `openai/clip-vit-base-patch32`, recorded revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, 512 dimensions, float32, L2 normalization, CPU, and batch size 64. Image embeddings cover 10,000/100/1,000 images; caption embeddings cover 50,000 raw training captions. QC validates finiteness, unit norms, and ID mappings. The text extraction statistics record elapsed time; available per-split image run metadata may not contain a comparable total wall-clock duration.

Separate FAISS `IndexFlatIP` indexes contain 10,000 training image vectors and 50,000 training caption vectors (cosine similarity because vectors are normalized). Index readback and mapping checks pass. Held-out caption IDs in the caption index: 0. Held-out image retrieval uses a train-only gallery; 20 queries, top-5, zero self matches, mean top-1 cosine 0.8260. Index build times and on-disk sizes are recorded in index metadata and the existing final report.

## Validation and reproducibility

Configuration is in `configs/config.json`; seed is 20261106. Historical selection evidence was unavailable, so a new geometry-only comparison of 224x224 letterbox, stretch, and center crop was performed on the existing 100 validation images. It records retained area, crop/padding, and aspect distortion, then selects letterbox because it preserves the full image and aspect ratio. It does not test downstream model performance and is not presented as a historical experiment. The environment/model versions are partially recorded; a complete unified environment snapshot remains absent.

## Compliance checklist

| Requirement | Status | Evidence / gap |
|---|---|---|
| Dataset selection and fixed split counts | PASS | `data/selected/*_ids.*`; QC manifest checks |
| Dataset integrity and split leakage | PASS | `reports/final_quality_control.json`; SHA-256 and pHash reports |
| Image decode and processed output integrity | PASS | QC decoded counts, no missing files; preprocessing report |
| EXIF orientation operation | PASS | `ImageOps.exif_transpose` in `src/pipeline.py` |
| EXIF correction counts / raw-to-output orientation verification | NOT_VERIFIABLE | Not recorded; raw images unchanged |
| RGB conversion and resize/letterbox | PASS | Pipeline source; 224×224 RGB outputs validated |
| Encoder normalization | PASS | CLIP processor; normalization config, not baked into JPEGs |
| Raw/clean caption separation | PARTIAL | Fields are separate; exact full source-caption mapping not covered by QC |
| Caption count and five-per-image cardinality | PASS | 55,500 records; QC checks each split |
| Caption vocabulary and length statistics | PASS | Documented tokenizer; 9,793 unique tokens, 591,664 tokens; per-split stats, frequency/token CSVs and visualizations |
| POS analysis | PASS | 55,500 captions; linguistic summary and QC |
| Object category/count/co-occurrence analysis | PASS | Statistical analysis CSVs and visualizations |
| Spatial relations | PASS | COCO-box-derived geometric relations in JSON/CSV and frequency chart; explicitly disclaimed as semantic labels |
| Visual scene/context/time analysis | PASS | 11,100 CLIP predictions; uncertainty and summary reports |
| Color and image-level visual proxies | PASS | Per-image coarse color/luminance/saturation and edge-density proxy JSON/CSV plus visualizations; no shape-recognition analysis claimed |
| Metadata extraction | PASS | Categories, boxes, counts, complexity, dimensions in `data/metadata/` |
| Image embeddings | PASS | QC shape, finite/unit norm, manifest-order checks |
| Caption embeddings | PASS | 50,000 raw train captions; mapping and vector checks |
| Train-only FAISS image/caption indexes | PASS | QC index counts, dimensions, readback, held-out exclusion |
| Retrieval anti-leakage | PASS | 20 held-out queries; zero self matches |
| Validation-based preprocessing selection | PASS (new geometry-only comparison) | Three configurations evaluated on existing 100 validation images; criteria, results, selection and performance limitation recorded; no historical experiment claimed |
| Reproducibility record | PARTIAL | Seed/model revisions and key versions recorded; unified environment/run record missing |
| Assignment PDF requirement mapping | PASS | Original PDF reviewed; detailed requirements specify 1,000 test images, with one isolated `1,00` typo |
| Test-size requirement | PASS | Frozen test split contains 1,000 images, consistent with the detailed assignment requirements |

## Remaining limitations and conclusion

The machine-readable QC report remains PASS for its pipeline checks, but that status is not equivalent to full assignment compliance. The four requested priority areas now have evidence-backed analyses. Remaining limitations include the incomplete environment snapshot, unrecorded historical EXIF/RGB-conversion counts, and partial exact source-to-caption mapping verification. The validation comparison is geometry-only; color, edge-density, and relation outputs retain their stated inference limits. Phase 1 compliance remains **PARTIAL**. The current 1,000-image test set and all frozen manifests remain intact.

The focused `python main.py extended_analysis` stage was run to regenerate only its reports, metadata and visualizations. It read raw images and captions plus existing annotation metadata; it did not change source images/captions, frozen manifests, embeddings or FAISS indexes. The existing QC report was inspected and remains PASS; model and index stages were not rerun.
