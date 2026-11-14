## Results (phase 1: preprocessing)

No captioning model has been evaluated yet. These are the results of the data preparation.
Caption-quality metrics, hallucination rates, statistical tests and ablations will be added
in phases 2 and 3.

### Summary

| Result | Value | Where |
|---|---|---|
| Quality-control audit | All checks passed | `reports/qc_report.json` |
| Retrieval sanity check (image to caption, first 1,000 KB images) | Recall@1 = 0.443, Recall@5 = 0.650 | `indexes/index_info.json` |
| Detector confidence threshold (chosen on validation) | 0.25 (F1 = 0.787) | `reports/dataset_stats_detected.json` |
| Embeddings | CLIP ViT-B/32, 512 dimensions, 10,000 image and 50,000 caption vectors | `embeddings/embedding_info.json` |
| Index | `IndexFlatIP` (exact), built in under 0.04 s | `indexes/index_info.json` |

The Recall values are an in-knowledge-base consistency check, not an evaluation on unseen
images.

### Detector threshold selection (100 validation images)

The classes found by YOLOv8n in each image were compared with the human COCO labels.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| **0.25** | 0.863 | 0.724 | **0.787** |
| 0.30 | 0.884 | 0.708 | 0.786 |
| 0.35 | 0.900 | 0.671 | 0.769 |
| 0.40 | 0.911 | 0.637 | 0.750 |
| 0.45 | 0.933 | 0.602 | 0.732 |
| 0.50 | 0.962 | 0.556 | 0.705 |
| 0.55 | 0.977 | 0.519 | 0.677 |
| 0.60 | 0.988 | 0.491 | 0.656 |
| 0.70 | 0.992 | 0.404 | 0.574 |

0.25 is the lowest value tested, because detections below it were not saved. 0.30 gives
almost the same F1.

### Objects per image

| Set | Detector at 0.25 | Detector at 0.5 | Human labels |
|---|---|---|---|
| Knowledge base (10,000) | 5.14 (160 images with none) | 3.22 (482 with none) | 7.21 (102 with none) |
| Validation (100) | 6.08 (0 with none) | 3.59 (7 with none) | |
| Test (1,000) | 5.05 (12 with none) | 3.10 (44 with none) | |

The detector finds fewer objects than the human annotators, who label every small object.

### Caption statistics (knowledge base, 50,000 captions)

| | Before cleaning | After cleaning |
|---|---|---|
| Average length | 10.47 words | 10.47 words |
| Vocabulary | 16,228 | 9,917 |

Part of speech: nouns 36.05%, determiners 20.48%, prepositions 15.96%, verbs 11.54%,
adjectives 7.90%, adverbs 0.65%. Captions rarely mention attributes.

Scene context (keyword estimate, images matching a caption keyword): 4,323 outdoor,
2,065 indoor, 448 day, 258 night.

### Figures (in `figures/`)

- Top-50 object classes (human labels and detected)
- Objects per image (human labels, and detector versus humans)
- Caption length, part-of-speech distribution
- Co-occurrence matrices (human labels and detected)
- Colour, shape and spatial words, scene context
- Detector threshold selection