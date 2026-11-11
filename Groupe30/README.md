# Group 30

## Students
- BOURABA Maroua
- BRAHIMI Yasmine
- DJILI Khaoula
- BOUAOUNE Maria Sara

## Project Title
A comparative study of RAG techniques for generating image captions: Metadata-Filtered RAG

## Submission Files
This submission covers **Phase 1 (data exploration and preprocessing)**. Where each item is:

| Item | Where |
|---|---|
| Source code | `notebooks/G30_Metadata_RAG_Phase1_Clean.ipynb`, `requirements.txt` |
| Dataset information | section **Dataset** and **Data splits** below, `data/` (split ID files, captions, YOLO metadata, manifests) |
| Results | `outputs/` (16 figures and two statistics files) and the sections **Exploration**, **Cleaning**, **Embeddings and search index** below |
| Final report | `report/Phase1_Data_Exploration_Preprocessing_FinalReport.pdf` |

---

## Phase 1: G30 Metadata-Filtered RAG for Image Captioning

Mini project, image processing. Group G30. Assigned technique: **Metadata-Filtered RAG** (retrieval filtered by visual metadata). Member profiles are in `team/`.

This repository contains **Phase 1: data exploration and preprocessing**. No captions are generated yet. The Phase 1 report is in `report/Phase1_Data_Exploration_Preprocessing_FinalReport.pdf`.

## The idea
A captioning model sometimes invents objects that are not in the photo (hallucination). With RAG, the model first reads captions of similar library photos. In our variant the objects of the new photo are detected first (YOLO), and only library photos that contain those objects are used as hints.

## Dataset
MS COCO 2017 (Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014): https://cocodataset.org/#download
- Photos: http://images.cocodataset.org/zips/train2017.zip and http://images.cocodataset.org/zips/val2017.zip
- Annotations and captions: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- The library (10,000) and validation (100) photos come from train2017; the 1,000 test photos come from val2017.
- This repository stores only the image IDs, captions, metadata and statistics of our subsets, not the photos.

## Data splits
| Split | Photos | Source | File |
|---|---|---|---|
| Knowledge base (library) | 10,000 | COCO train2017 | `data/splits/kb_ids.json` |
| Validation | 100, different from the 10,000 | COCO train2017 | `data/splits/val_ids.json` |
| Test | 1,000 | COCO val2017 | `data/splits/test_ids.json` |

- How they were drawn: the image IDs of train2017 and of val2017 were shuffled with Python's `random` (seed 42). The library is the first 10,000 shuffled train2017 IDs, the validation set the next 100, and the test set the first 1,000 shuffled val2017 IDs (each list sorted).
- Library plus validation make 10,100 train2017 photos.
- `data/splits/test_ids_100.json` is a second list of 100 test photos, selected from the 1,000 by notebook cell B4 (30 simple, 40 medium and 30 complex scenes, by number of COCO-labelled objects). It is kept for comparison and is not used elsewhere in this repository.

**Anti-leakage checks** (`data/processed/caption_audit.json`, `data/splits/cleaning_report.json`):
- no photo appears in two splits (library and validation: 0, library and test: 0, validation and test: 0);
- no validation or test caption appears in the library captions (overlap: 0);
- no photo in one split has the same perceptual image hash as a photo in another split (0);
- validation and test photos are not in the search index.

## Cleaning
Checked on the 11,100 photos of the three splits (`data/splits/cleaning_report.json`):

| Check | Result |
|---|---|
| Missing photos | 0 |
| Corrupted photos | 0 |
| Exact duplicates | 0 |
| Near-duplicates (same perceptual hash) | 0 |
| Photos with fewer than 5 captions | 0 |
| Photos with 6 captions | 35 (31 library, 1 validation, 3 test) |

The 10,000 library photos have **50,031** captions. Nothing was removed.

## Captions
- Each library caption is stored twice in `data/processed/kb_captions.jsonl`: `caption_raw` (unchanged, for evaluation) and `caption_clean` (lowercased, punctuation removed, spaces normalised, for indexing).
- The raw human captions of the validation and test photos are in `data/processed/eval_references_val.json` and `eval_references_test.json`. They are not indexed.

## Objects extracted from the images (YOLO)
- Objects are detected directly in the images with YOLOv8n (`yolov8n.pt`, Ultralytics 8.4.197), with the default settings: confidence threshold 0.25, image size 640. The same settings were used for the library, validation and test photos (`data/processed/yolo_settings.json`).
- `data/processed/yolo_metadata.json`: library and validation photos (10,100). `data/processed/yolo_metadata_test.json`: test photos (1,000). Each record has the object names and the number of detections.
- Photos with no detected object: 177 library, 5 validation, 22 test.
- Each record of `data/index/kb_records.json` carries the YOLO object labels of its photo.
- The human COCO labels are used only for the "before" exploration below. The objects of the test photos come from the detector, not from their COCO labels.

## Embeddings and search index
| Item | Value |
|---|---|
| Model | CLIP `openai/clip-vit-base-patch32`, for photos and for captions (CLIP text encoder) |
| Vectors | 512 dimensions, float32, L2-normalised |
| Counts | 10,000 library photos, 50,031 library captions, 100 validation photos, 1,000 test photos |
| Generation | GPU, batch size 64, torch 2.11.0+cu130, transformers 5.21.0; no failed images |
| Time | library photos 104.4 s, library captions 33.7 s |
| Index | FAISS 1.15.1, `IndexIDMap2(IndexFlatIP)` (exact search), cosine similarity |
| Index size | photos 20.56 MB, captions 102.86 MB |
| Indexing time | 0.088 s |
| Checks | self-retrieval@1 = 1.0, image-to-caption hit@5 = 0.666 |

See `data/embeddings/embedding_manifest.json` and `data/index/kb_manifest.json`. `data/index/caption_to_image.json` links each caption vector to its photo.

## Exploration (library, 10,000 photos)
**Before preprocessing:** human COCO labels and raw captions (figures 01 to 10, `outputs/phase1_statistics.json`).
**After preprocessing:** YOLO objects and cleaned captions (figures 11 to 16, `outputs/phase1_statistics_after.json`).

| Measure | Before | After |
|---|---|---|
| Photos with a person | 53.98% | 52.15% |
| Mean objects per photo | 7.27 | 5.49 |
| Median objects per photo | 4 | 4 |
| Photos with no object | 96 | 177 |
| Simple scenes (1 to 3 objects) | 4,143 | 4,575 |
| Medium scenes (4 to 7) | 2,456 | 2,906 |
| Complex scenes (8 or more) | 3,305 | 2,342 |
| Mean caption length (words) | 10.45 | 10.46 |
| Vocabulary size | 9,864 | 9,762 |

- Most frequent detected classes, by number of photos: person (5,215), car (918), dining table (917), chair (852), cup (660).
- Object classes in the human labels: 80. Caption length: standard deviation 2.40 words.
- Most frequent colour words (captions containing them): white (2,996), black (1,617), red (1,394). Keyword counts: 9,859 captions with outdoor words, 6,838 with indoor words (heuristic).
- Spatial relations between the 8 largest labelled objects of each photo (human boxes): overlapping 40,411, left of 33,562, right of 33,144, below 11,892, above 7,259.
- The test photos have a scene mix close to the library's (YOLO-based, figure 16): simple 45%, medium 30%, complex 23% on the test set, against 46%, 29%, 23% on the library.

## Notebook
`notebooks/G30_Metadata_RAG_Phase1_Clean.ipynb`
- **Part A (A1 to A4):** restores the project files from the zip on Google Drive and runs an integrity audit. The last line must read `ALL CHECKS PASSED`.
- **Part B:** cleaning report (B1), YOLO on the test photos (B2), exploration before preprocessing (B3), exploration after preprocessing (B6), an optional cell that saves every detection with its confidence and box (B7), spatial relations from those boxes (B8), and the filling of the object labels in `kb_records.json` (B9). B4 and B5 are optional tools for a 100-photo test list.
- **Part C:** packaging of the results.

To run it: put the zip in My Drive (right-click the shared file, then **Make a copy**), set `ZIP_NAME` in the CONFIG cell to the zip name (`RAG_Project_Phase1_Light.zip` for Part A, `RAG_Project_Phase1_Fixed_Submission.zip` for Part B), and run the cells from top to bottom, one at a time. Part B needs the full archive, which contains the photos.

## Repository structure
```
notebooks/        the notebook
data/splits/      frozen ID files and the cleaning report
data/processed/   captions, YOLO metadata, evaluation references, caption audit
data/embeddings/  embedding manifest
data/index/       index manifest, caption-to-image links, library records
outputs/          figures and statistics
team/             member profiles
report/           Phase 1 report (PDF)
```

## Files not stored here
The photos, the embeddings (`kb_image_embeddings.npy`, `kb_caption_embeddings.npy`, `val_image_embeddings.npy`, `test_image_embeddings.npy`) and the FAISS indexes (`kb_image.index`, `kb_caption.index`) are too large for GitHub. They are kept in zip archives on the team's Google Drive. 

- **Light archive** (embeddings, FAISS indexes, metadata and reports, no photos; used by notebook Part A): https://drive.google.com/file/d/1oB4GNurCtXfXjahL5qZfPr145FnzlzB6/view?usp=drive_link
- **Full archive** (also contains the COCO photos and annotation files, over 2 GB; needed for notebook Part B): https://drive.google.com/file/d/1MTSCuB7bfAB1i1y39dz9uA_zQSkiPzuj/view?usp=sharing
- The photos can also be downloaded directly from COCO (see Dataset).

## Known limitations
- The test set is a random sample from val2017, not stratified by scene type; its scene mix is close to the library's (figure 16).
- YOLOv8n detects fewer objects than the human labels (mean 5.49 against 7.27 per photo, and 2,342 against 3,305 complex scenes). 177 library photos have no detected object.
- The scripts that generated the embeddings and the index are not included in this repository. Their settings are recorded in the manifests.
- The indoor, outdoor, day and night counts are keyword heuristics, not a scene classifier.
- 35 photos have 6 captions instead of 5; the extra caption is kept.

## Next phases
Phase 2 (model training and validation) is due on 2 December 2026, and Phase 3 (model evaluation) on 21 December 2026.
