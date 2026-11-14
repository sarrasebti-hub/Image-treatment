# RAG for image captioning: preprocessing (phase 1)

Shared preprocessing for a comparative study of RAG techniques on MS COCO.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Download the COCO 2017 annotations (annotations_trainval2017.zip) from
cocodataset.org 

## Pipeline (run in this order)
1.  explore.py             quick look at the annotation files
2.  split.py               knowledge-base split (10,000 train2017 images, seed 42)
3.  validation.py          validation (100) and the first test set (100)
4.  split_test1000.py      test set of 1,000 val2017 images (contains the 100)
5.  images.py              downloads the imagesgit add .
6.  stats.py               statistics, before preprocessing
7.  preprocess.py          raw + clean captions
8.  embeddings.py          CLIP embeddings
9.  build-index.py         FAISS indexes
10. extract_metadata.py    object labels from COCO (human, confidence = 1)
11. detect_objects.py      objects detected in the images (YOLO)
12. stats2.py              statistics from captions and human labels
13. stats3.py              threshold selection, detected-object statistics, before/after captions
14. audit.py               quality control and leakage audit

## Splits
| Split | Source | Size |
|---|---|---|
| Knowledge base | COCO train2017 | 10,000 images, 5 captions each |
| Validation | train2017, not in the knowledge base | 100 |
| Test | COCO val2017, stratified by object count | 100 |

Evaluation captions are never in the knowledge base (checked in `audit.py`).

## Results of phase 1
- Embeddings: CLIP ViT-B/32, 512-d, L2-normalised
- Index: IndexFlatIP (exact), cosine similarity
- Image-to-caption sanity check: Recall@1 = 0.443, Recall@5 = 0.650
- Quality control: all checks passed


📄 **[Click here to view the Project Report](Phase1_Report_Final.pdf)**