# Dataset Sources

## Original Dataset
- **Name:** MS COCO 2017 (Microsoft Common Objects in Context)
- **Website:** https://cocodataset.org
- **Splits used:** `train2017` (knowledge base and validation images) and `val2017` (test images)

## Annotation Files Used
| File | Content |
|---|---|
| `instances_train2017.json` | Object annotations and bounding boxes (train) |
| `captions_train2017.json` | Captions (train) |
| `captions_val2017.json` | Captions (val), used for the test set |

## Test Images
The 100 test images were downloaded from a Hugging Face mirror, because the original COCO image server returned an SSL hostname verification error. The files were then checked to confirm they are valid JPEG images.

## Project Data (Google Drive)
The selected images, annotations, metadata and statistics are stored in a shared Google Drive folder, `COCO_Project`:

**Link:** PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE

The images are not uploaded to this repository because of GitHub size limits.

## Models
- **CLIP:** `openai/clip-vit-base-patch32` (Hugging Face Transformers), used for image preprocessing and for image and text embeddings.
