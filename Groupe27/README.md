# A Comparative Study of RAG Techniques for Image Captioning (Group 27)

- **University:** Université d'Alger 1 Benyoucef Benkhedda, Département Informatique
- **Module:** Introduction to Image Processing
- **Supervisor:** SEBTI Sarra
- **Group 27:** Younes Harhour, Ryad Boudenne, Mazarine Keramane, Lyna Slimani
- **Assigned technique:** RAG query expansion (groups G25 to G27)

This repository contains Phase 1 of the project: data exploration, cleaning, splitting and the embedding knowledge base used for retrieval in the next phases. The Phase 1 report is in [`report/Image_processing_phase_1.pdf`](report/Image_processing_phase_1.pdf).

## Phase 1 results

| Step | Result |
|---|---|
| Cleaning audit | 123,287 annotated COCO train2017 + val2017 images checked: 15 exact duplicates removed, 0 corrupt, 0 missing |
| Knowledge base | 10,000 images from train2017, 50,036 captions |
| Validation | 100 images drawn from within the knowledge base |
| Test | 100 images from val2017, no overlap with the knowledge base (by ID, file hash and pixel hash) |
| Captions | 10.5 words on average (median 10), vocabulary of 9,615 words |
| Scene complexity | 7.27 annotated objects per image over all 10,000 images (7.34 over the 9,907 images with at least one object), all 80 COCO categories present |
| Embeddings | CLIP ViT-B/32 image and caption vectors (512-d), MiniLM-L6-v2 caption vectors (384-d), exact FAISS indexes (`IndexFlatIP`, cosine) |
| Quality control | 20/20 checks passed, anti-leakage audit passed |

Splits were drawn by uniform random sampling with seed 42 and are frozen for the rest of the project.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── clean_and_split.py          # cleaning audit of the raw COCO images and split creation
├── verify_and_visualize.py     # re-checks the cleaned dataset and draws the dataset charts
├── build_embeddings.py         # CLIP + Sentence-BERT embeddings and FAISS indexes
├── check_embeddings.py         # embedding quality control and anti-leakage audit
├── retrieval.py                # KnowledgeBase class used to query the indexes
├── object_counting.ipynb       # objects-per-image analysis (run on Google Colab)
├── notebooks/
│   └── 01_caption_analysis.ipynb   # caption length, vocabulary, POS, attributes, context, Zipf
├── cleaned_dataset/            # frozen splits, annotations, audit files, charts (see its README)
├── embeddings/                 # ID maps, caption table, metadata, QC report (see its README)
├── figures/
│   ├── caption_analysis/       # figures from the caption analysis notebook
│   └── object_counting/        # figure from the object counting notebook
├── outputs/
│   └── kb_captions_clean.csv   # raw and cleaned knowledge-base captions
└── report/
    └── Image_processing_phase_1.pdf
```

More detail on each part:
- [`cleaned_dataset/README.md`](cleaned_dataset/README.md) and [`cleaned_dataset/DELIVERABLES.md`](cleaned_dataset/DELIVERABLES.md): cleaning, splits and charts.
- [`embeddings/README.md`](embeddings/README.md): encoders, preprocessing, index settings, timings, leakage audit and how to query the knowledge base.

## Data

Large files are not stored on GitHub:

- **Images** (10,200 files, about 1.7 GB): download the `cleaned_dataset` folder from Google Drive and copy each split's `images/` folder into the matching folder here (`cleaned_dataset/knowledge_base/images/`, `cleaned_dataset/validation/images/`, `cleaned_dataset/test/images/`).
  https://drive.google.com/drive/folders/1xh4MCusgpVPTY2YH83h9KwDsod2Iqn36
- **Embeddings and FAISS indexes** (about 380 MB): rebuild them with `build_embeddings.py` (see below). `check_embeddings.py` compares the result with the SHA256 checksums stored in `embeddings/metadata.json`.

Everything else (split IDs, captions, object annotations, manifests, audit files, statistics, figures) is in this repository.

## Installation

Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For GPU encoding, install the CUDA build of PyTorch from https://pytorch.org before `requirements.txt`. The embeddings were built with torch 2.6.0+cu124 on an RTX 3060; the CPU works too, only slower.

## Reproducing Phase 1

Run the commands from the repository root.

1. **Check the dataset and redraw the charts** (needs the images from Google Drive):
   ```bash
   python verify_and_visualize.py
   ```
   Writes `cleaned_dataset/verification.json` and the charts in `cleaned_dataset/visualizations/`.

2. **Caption analysis:** open `notebooks/01_caption_analysis.ipynb`. Part 1 analyses the full COCO train2017 captions and needs `data/captions_train2017.json` (from `annotations_trainval2017.zip` on https://cocodataset.org). Part 2 runs on the knowledge base only and needs nothing else.

3. **Object counting:** `object_counting.ipynb` was run on Google Colab. To run it locally, set the `dossier` variable to `cleaned_dataset/knowledge_base`.

4. **Embeddings and quality control:**
   ```bash
   python build_embeddings.py
   python check_embeddings.py
   ```
   The first run downloads about 700 MB of model weights (pinned versions). `build_embeddings.py` refuses to overwrite an existing release unless `--overwrite` is passed.

5. **Full cleaning from the raw COCO data** (optional): `clean_and_split.py` re-runs the complete audit. It needs the original COCO train2017/val2017 images and caption annotations in the folder layout recorded in the script, and an empty output location; it will not overwrite an existing `cleaned_dataset` folder.

## Anti-leakage rules

- Test captions are evaluation references only. They are never embedded or indexed, and test images are not in any index.
- Validation images are part of the knowledge base, so a validation query must exclude its own image and captions (`exclude=[{image_id}]` in `retrieval.py`).
- The split IDs in `cleaned_dataset/*/image_ids.json` must not be changed.
