# Dataset Audit, Cleaning and Splitting Report

## 1. Project Information

- **Project:** RAG-Based Image Captioning
- **Dataset:** MS COCO 2017
- **Responsible:** Chahinez
- **Assigned Task:** Data Audit, Cleaning and Splitting

## 2. Original Dataset Audit

The original COCO Train2017 dataset was audited.

| Check | Result |
|---|---:|
| Total images | 118,287 |
| Total captions | 591,753 |
| Missing images | 0 |
| Extra image files | 0 |
| Corrupted images | 0 |
| Exact duplicate image files detected | 14 |

**Duplicate handling:** 14 duplicate files were detected during the original dataset audit. They were not explicitly removed from the original dataset.

## 3. Caption Verification

| Captions per image | Number of images |
|---|---:|
| Exactly 5 captions | 117,972 |
| 6 captions | 312 |
| 7 captions | 3 |

All audited images had at least five captions.

## 4. Dataset Splitting

A reproducible random selection using **seed 42** was used.

| Split | Source | Images |
|---|---|---:|
| Knowledge Base | Train2017 | 10,000 |
| Validation | Train2017 | 100 |
| Test | Val2017 | 100 |
| **Total** | | **10,200** |

The Test subset was sampled proportionally from three groups based on the number of object annotations per image:

- Simple: up to 3 objects.
- Medium: 4–10 objects.
- Complex: more than 10 objects.

This grouping is based on object counts and does not represent every aspect of scene complexity.

## 5. Processed Dataset Verification

| Split | Images | Captions |
|---|---:|---:|
| Knowledge Base | 10,000 | 50,028 |
| Validation | 100 | 500 |
| Test | 100 | 500 |
| **Total** | **10,200** | **51,028** |

The final verification confirmed:

- No missing image files.
- No extra image files.
- No corrupted processed images.
- No invalid caption references.
- No images without captions.
- No overlapping image IDs between splits.

An additional exact-file duplicate check found **zero duplicate groups** among the 10,200 selected processed images. This check does not rule out visually similar images or semantic duplicates.
## 6. Final Dataset State After Test Expansion

The original test split contained 100 images. It was expanded by adding 900 additional images from the official COCO Val2017 dataset, resulting in 1,000 test images.

The original 100-image metadata and captions files were backed up before expansion.

### Final Split Statistics

| Split | Images | Captions |
|---|---:|---:|
| Knowledge Base | 10,000 | 50,028 |
| Validation | 100 | 500 |
| Test | 1,000 | 5,002 |
| **Total** | **11,100** | **55,530** |

### Data Leakage Verification

The three splits were checked using image IDs.

- Knowledge Base vs Validation: 0 overlapping images.
- Knowledge Base vs Test: 0 overlapping images.
- Validation vs Test: 0 overlapping images.

**Result:** No image ID overlap was detected between the three splits.

### Test Split Integrity

- Total test images: 1,000.
- Unique image IDs: 1,000.
- Total captions: 5,002.
- Images without captions: 0.
- Captions linked to unknown images: 0.
- Duplicate annotation IDs: 0.

The final dataset contains 10,000 Knowledge Base images, 100 Validation images, and 1,000 Test images.
