# Dataset Description

## Overview
This project uses **MS COCO 2017** (Microsoft Common Objects in Context), a dataset widely used in computer vision. It contains images of everyday objects and real-world scenes, with object annotations (categories and bounding boxes) and human-written captions. The dataset has 80 object categories.

## Original Dataset (train2017 annotations)
| Element | Quantity |
|---|---|
| Images | 118,287 |
| Object annotations | 860,001 |
| Object categories | 80 |
| Caption annotations | 591,753 |

## Subsets Used in the Project
| Subset | Images | Source | Purpose |
|---|---|---|---|
| Knowledge Base | 10,000 | train2017 | Main collection for image and text retrieval |
| Validation | 100 | train2017 | Validation and pipeline testing |
| Test | 100 | val2017 | Independent evaluation (500 reference captions) |
| **Total** | **10,200** | | |

Images were selected randomly with a fixed seed of 42. There is no overlap between the three subsets.

## Statistics of the Selected Training Data (10,100 images)
- **Captions:** 50,521 (mean length 10.46 words, median 10, min 6, max 47)
- **Object annotations:** 72,752, covering all 80 categories (person is the most frequent with 22,319)
- **Objects per image:** mean 7.20, median 4, max 56
- **Orientation:** 7,328 landscape, 2,398 portrait, 374 square
- **Image size:** width and height up to 640 pixels (median 640 × 480)
- **Images without object annotations:** 104 (kept, since each has five captions)

## Quality Checks
No missing images, no corrupted images, no exact duplicates, and no overlapping image IDs were found.

## Preprocessing
Images are processed with the CLIP preprocessor of `openai/clip-vit-base-patch32`: RGB conversion, resizing (shortest edge 224), center crop 224 × 224, rescaling and normalization. Original images are never modified.

## Metadata Files
See the `metadata/` folder for image identifiers, manifests, the dataset audit report and the CLIP preprocessing configuration.
