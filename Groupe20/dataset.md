## Dataset general information

**The Source:** MS COCO 2017 (images and 5 human-written captions per image). The annotations
are released under CC BY 4.0, and the images come from Flickr under their own licenses.

### Splits (fixed with seed 42, IDs stored in `splits/`)

| Split | Source | Size | Used for |
|---|---|---|---|
| Knowledge base | COCO train2017 | 10,000 images, 50,000 captions | What the RAG system retrieves from |
| Validation | train2017, not in the knowledge base | 100 images, 500 captions | Tuning settings and the detector threshold |
| Test | COCO val2017, stratified by object count (simple, medium, complex scenes) | 1,000 images, 5,000 captions | Final evaluation |
| Test (small) | Subset of the 1,000 test images | 100 images | Comparison with the full test set |

Images with fewer than 5 captions are excluded. Evaluation images and their captions are
never part of the knowledge base, and a test image can never be retrieved for itself
(checked by `audit.py`).

### Captions
- `caption_raw`: original COCO captions, exactly 5 per image, used for evaluation.
- `caption_clean`: lowercase, no punctuation, used only for retrieval. Validation and test
  images have no cleaned copy.
- Files: `processed/kb_captions.json`, `val_references.json`, `test_references.json`
  (100 images) and `test_1000_references.json` (1,000 images).
- Before cleaning, the knowledge-base captions have 16,228 distinct words. After cleaning,
  they have 9,917. The average length is 10.47 words in both cases.

### Embeddings and index
- CLIP ViT-B/32, 512 dimensions, L2-normalised: 10,000 image vectors and 50,000 caption
  vectors.
- FAISS `IndexFlatIP` (exact search, inner product equals cosine). Details are in
  `indexes/index_info.json`.
- Sanity check (image to caption, first 1,000 knowledge-base images): Recall@1 = 0.443,
  Recall@5 = 0.650. This is a consistency check inside the knowledge base, not an
  evaluation on unseen images.

### Object metadata
- **Detected objects:** YOLOv8n run on all 11,100 images (10,000 knowledge base, 100
  validation, 1,000 test). Every detection above confidence 0.25 is stored with its class,
  confidence and box in `metadata/detected_kb.json`, `detected_val.json` and
  `detected_test1000.json`. The confidence threshold of 0.25 was chosen on the validation
  images (F1 = 0.787 against the human labels).
- **Human labels:** COCO instance annotations (confidence 1.0) in `metadata/kb_metadata.json`,
  `val_objects_gt.json` and `test_objects_gt.json`.
- Knowledge-base objects per image: 7.21 for the human labels, 5.14 for the detector.

### Statistics
Exploration before preprocessing (raw captions, human labels) and after (cleaned captions,
detected objects). The graphs are in `figures/` and the numbers in `reports/`.

### Not stored in the repository
Images (`images/`), embeddings (`embeddings/*.npy`), indexes (`indexes/*.index`) and the
detector weights (`yolov8n.pt`) are too large or are downloaded automatically. Rebuild them
by running the pipeline in the order given above (`images.py`, `embeddings.py`,
`build-index.py`, `detect_objects.py`).

### Quality control
`audit.py` runs the automatic checks: split sizes, no overlap between splits, no test image
in the index, five captions per image, no corrupt images, no broken embeddings, and valid
detection files. Result: all checks passed (`reports/qc_report.json`).