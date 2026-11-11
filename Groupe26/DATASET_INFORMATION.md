# Dataset Information

## Where the images came from

The images come from the **Microsoft COCO 2017** dataset, specifically the `train2017` and `val2017` image archives. The source dataset and annotations are available from the [official COCO download page](https://cocodataset.org/#download). The notebook used the local dataset root `E:\projet\archive` (configurable with the `COCO_ROOT` environment variable).

The local folders were `train2017\train2017` and `val2017\val2017`. COCO 2017 instance annotations and human-written captions were read from the matching annotations archive. The selected images were not taken from a third-party or user-created image set.

The complete source collection contained **118,287 train2017 images** and **5,000 val2017 images**.

## How many images were used and how they were split

All source images were checked before the project splits were selected. The notebook sorts each source file list, shuffles it with NumPy's fixed random seed (`42`), verifies files and removes exact duplicates from the candidate pool, then selects the requested counts. The validation and test images are selected from their respective COCO source splits, not by splitting captions or annotations.

| Project split | COCO source | Image count | Purpose |
|---|---|---:|---|
| Knowledge base | `train2017` | 10,000 | Image retrieval database and object extraction |
| Project validation | `train2017` | 100 | Development and quality evaluation; separate from the knowledge base |
| Test | `val2017` | 1,000 | Final evaluation, separate from the training source |
| Comparison test | First 100 images of the test split | 100 | Keeps the smaller test set for comparison |

The knowledge base and project validation set are disjoint, for **10,100 selected `train2017` images** in total. The test split uses a further 1,000 images from COCO `val2017`. The 100-image comparison set is the first 100 entries of that test split, so it is included in (not added to) the 1,000 test images.

The exact filenames and selection are saved in [`outputs/phase1/frozen_splits.json`](./outputs/phase1/frozen_splits.json). With the same source files and notebook, seed `42` reproduces the selection.

## What was done to the images and data

1. **File verification:** every JPEG was opened and checked for image integrity.
2. **Duplicate detection:** SHA-256 hashes were compared across the source splits. One representative of each byte-identical image was retained in the clean candidate pool.
3. **Annotation integrity:** COCO annotation filenames were checked against files on disk.
4. **Split selection:** the clean, unique candidates were shuffled and selected into the knowledge base, project validation, and test sets described above.
5. **Metadata and exploration:** COCO captions and instance annotations were linked to image IDs. Caption length, object category counts, and per-image scene complexity were measured.
6. **Object extraction:** YOLOv8n detected objects in the 10,000 knowledge-base images using a confidence threshold of `0.25`. Labels and detector confidences were saved per image.
7. **Retrieval preparation:** detected labels were converted into text descriptions, embedded with `all-MiniLM-L6-v2`, and indexed using FAISS.

### Image resizing

The original COCO JPEGs were **not resized, overwritten, or re-encoded**; no resized image copies were created by this pipeline. For YOLO inference only, Ultralytics uses a default target size (`imgsz`) of **640 pixels**. It resizes/letterboxes images in memory for detection, preserving their aspect ratio with padding as needed. This model-input resizing does not change the source images in the COCO folders.

Captions and annotations remain the original COCO data. Each selected image has five human-written reference captions. They are retained as ground-truth references and are not rewritten by the preprocessing steps.

## Cleaning and integrity results

| Source split | Files checked | Corrupt files | Exact duplicates | Valid unique images |
|--------------|--------------:|--------------:|-----------------:|--------------------:|
|  `train2017` |    118,287    |       0       |        14        |       118,273       |
|  `val2017`   |      5,000    |       0       |        0         |         5,000       |

No annotation-linked image files were missing from disk. Source images are not deleted or modified; duplicate handling affects the selected candidate lists only. See [`outputs/phase1/cleaning_audit.json`](./outputs/phase1/cleaning_audit.json) and [`outputs/phase1/train_missing_image_files.json`](./outputs/phase1/train_missing_image_files.json) for audit details.

The duplicate check identifies byte-identical images; it does not detect visually similar images saved with different file contents.

## Dataset contents and metadata

COCO supplies:

- **Object annotations:** labeled object instances from the COCO category set, used for the pre-processing category and scene-complexity statistics.
- **Captions:** human-written image descriptions, used as reference captions for the selected knowledge-base, validation, and test images.
- **Image IDs and filenames:** used to link each image with its annotations and captions.

Every selected image has five human-written reference captions. These captions are treated as ground truth, with reference confidence **1.0**. Caption references are stored separately by split; test references are not used to construct the knowledge-base embeddings.

## Project-derived image metadata

YOLOv8n object detection is run on all 10,000 knowledge-base images. The exported metadata records each filename, COCO image ID, detected object labels, detection confidence scores, extraction model, and detector threshold (**0.25**). Detector confidence is separate from the confidence assigned to human-written captions.

The per-image detections are saved in [`outputs/phase1/extracted_metadata_yolo.json`](./outputs/phase1/extracted_metadata_yolo.json). The pipeline generated 52,180 detections across 79 classes; the average was 5.22 detections per image (median 4), with 156 images having no detections.

## Dataset statistics

COCO `train2017` annotation statistics before restricting to the project knowledge base:

- 591,753 captions.
- Average caption length: 10.46 words.
- Vocabulary size after basic punctuation cleanup: 30,052 whitespace-tokenized words.
- Average annotated object instances per image: 7.27; median: 4.
- Most frequent annotated categories by instance count: person (262,465), car (43,867), chair (38,491), book (24,715), bottle (24,342), cup (20,650), dining table (15,714), bowl (14,358), traffic light (12,884), and handbag (12,354).

Statistics and visualization files are available in [`outputs/phase1/preprocessing_statistics.json`](./outputs/phase1/preprocessing_statistics.json), [`outputs/phase1/train_image_statistics.csv`](./outputs/phase1/train_image_statistics.csv), and the before/after preprocessing charts in [`outputs/phase1`](./outputs/phase1).

## Embeddings

For retrieval, each knowledge-base image's detected object labels are converted into text and encoded with `all-MiniLM-L6-v2`. The saved embedding matrix contains 10,000 normalized 384-dimensional vectors; the FAISS inner-product index contains the same 10,000 records. These are embeddings of object-description text linked to images, not embeddings of raw image pixels.

Files: [`outputs/phase1/kb_embeddings.npy`](./outputs/phase1/kb_embeddings.npy) and [`outputs/phase1/faiss_kb.index`](./outputs/phase1/faiss_kb.index).

## Reproducibility

See the [Phase 1 notebook](./phase 1.ipynb) for the cleaning, splitting, exploration, extraction, and embedding pipeline. Run the notebook cells from top to bottom. The split seed and detector threshold are defined in the notebook, and the frozen split manifest records the exact filenames used.
