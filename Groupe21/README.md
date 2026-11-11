# A Comparative Study of RAG Techniques for Image Captioning (MS COCO)

> **Group:** 3 and 4
> **Assigned Technique:**  
> **Members:**
>
> **Mini-Project – Image Processing**  
> **Current Phase:** Phase 1 – Data Exploration and Preprocessing  
> **Phase 1 Deadline:** 4 November 2026

## 1. Project Goal

This project aims to study how different Retrieval-Augmented Generation (RAG) techniques influence the quality, relevance, and visual grounding of image captions generated on a subset of the MS COCO dataset.

Phase 1 focuses on building clean, frozen, and shared data artifacts, including dataset splits, caption tables, metadata, embeddings, and a vector index. These artifacts will be used in the subsequent experimental phases.

The project is divided into three phases:

- **Phase 1:** Data exploration and preprocessing.
- **Phase 2:** Model training, validation, and RAG implementation.
- **Phase 3:** Evaluation and comparison of the different techniques.

## 2. Dataset and Data Splits

The project uses the **MS COCO 2017** dataset.

### 2.1 Data Splits

| Split               | Source                 | Images | Captions | Role                  |
| ------------------- | ---------------------- | -----: | -------: | --------------------- |
| Knowledge Base (KB) | COCO train2017         | 10,000 |   50,000 | Retrieval corpus      |
| Validation          | Subset of KB (seed 42) |    100 |      500 | Hyperparameter tuning |
| Test                | COCO val2017           |    100 |      500 | Final evaluation      |

All image IDs are stored in `splits/splits_all.json`. The splits are frozen and must not be modified after finalization.

The test set is stratified into five groups of 20 images:

- Simple scenes (1–3 objects).
- Complex scenes (10 or more objects).
- Multi-category scenes (5 or more categories).
- Interactions (person + other objects).
- Varied contexts.

Sampling seed: `42`.

### 2.2 Anti-Leakage Rules

To ensure reliable evaluation:

1. The Knowledge Base and Test sets are disjoint.
2. Test captions are never embedded or indexed. They are stored separately in `processed/test_captions.csv`.
3. Validation retrieval must exclude the query image itself because validation images are part of the Knowledge Base.
4. Test data is reserved exclusively for final evaluation.

## 3. Repository Structure

```text
image_processing/
│
├── README.md
├── .gitignore
│
├── notebook/
│   ├── one.ipynb
    ├── data_exploration.ipynb
│   ├── preprocessing_captions.ipynb
│   ├── preprocessing_images.ipynb
│   └── splits_audit.ipynb
│── notebook/data
│   ├── caption_statistics.csv
    ├── object_statistics.csv
│   ├── scene_statistics.csv
│   ├── top_words.csv
│   └── top50_categories.csv
│
│
│
├── splits/
│   └── splits_all.json
│
├── data/                 # Raw COCO images and annotations (git-ignored)
│
├── processed/            # Generated processed data
├── figures/               # Exploration figures
├── reports/               # Analysis and audit reports
├── embeddings/             # Generated embeddings (git-ignored)
└── index/                  # Vector indexes (git-ignored)
```

Some folders are generated during the preprocessing and experimentation stages.

## 4. Installation

### 4.1 Requirements

- Python 3.10 or later.
- Jupyter Notebook.
- VS Code

### 4.2 Environment Setup

Clone the repository:

```bash
git clone https://github.com/Ferielhmn/image_processing.git
cd image_processing
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow
```

Additional libraries may be required for the embedding and RAG stages.

## 5. Phase 1 – Data Exploration and Preprocessing

### 5.1 Image Preprocessing

The image preprocessing stage includes:

- Loading and inspecting images.
- Checking image dimensions and formats.
- Detecting missing or corrupted images.
- Converting grayscale images to RGB when necessary.
- Correcting image orientation using EXIF metadata.
- Preparing images for future model processing.

### 5.2 Caption Preprocessing

The caption preprocessing stage includes:

- Loading image captions.
- Cleaning text data.
- Converting captions to lowercase.
- Removing unnecessary punctuation and apostrophes.
- Tokenization.
- Calculating caption lengths.
- Analyzing vocabulary.
- Detecting empty and duplicated captions.

### 5.3 Data Exploration

The exploration stage includes:

- Caption length distribution.
- Vocabulary analysis.
- Most frequent words.
- Number of objects per image.
- Number of categories per image.
- Object category distribution.
- Image and caption diversity.

## 6. Preprocessing Summary

| Step                     | Decision                                                                                                             |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Image audit              | 10,100 images verified using `PIL.Image.verify()`.                                                                   |
| Missing images           | 14 images were missing after the first download and were re-downloaded. Final result: 0 missing or corrupted images. |
| Duplicates               | 0 duplicated IDs and 0 duplicated images (MD5).                                                                      |
| Image mode               | 16 grayscale KB images and 1 test image converted to RGB.                                                            |
| EXIF                     | Orientation corrected using `ImageOps.exif_transpose`.                                                               |
| Resize and normalization | Handled by `CLIPProcessor` (224 resize, center crop, CLIP mean/std normalization).                                   |
| Captions                 | Five captions retained per image. Empty captions: 0.                                                                 |
| Duplicate captions       | Kept and flagged using `is_duplicate`.                                                                               |
| Raw and clean captions   | `caption_raw` for evaluation and `caption_clean` for retrieval.                                                      |
| Metadata                 | Objects, categories, supercategories, bounding boxes, and areas extracted from COCO annotations.                     |

## 7. Knowledge Base – Embeddings and Vector Index

The planned knowledge base uses the following components:

| Component       | Technology        | Description                                    |
| --------------- | ----------------- | ---------------------------------------------- |
| Image Encoder   | CLIP ViT-B/32     | `openai/clip-vit-base-patch32`, 512 dimensions |
| Text Encoder    | Sentence-BERT     | `all-MiniLM-L6-v2`, 384 dimensions             |
| Vector Database | FAISS             | `IndexFlatIP` for exact similarity search      |
| Similarity      | Cosine Similarity | Inner product on L2-normalized vectors         |

Index size and processing time will be documented in the corresponding experiment reports.

## 8. Exploration Results

The exploration stage focuses on:

- Top 50 object classes.
- Objects per image.
- Categories per image.
- Spatial relationships.
- Class co-occurrence.
- Caption length distribution.
- Vocabulary analysis.
- Part-of-speech distribution.
- Color and shape diversity.
- Indoor and outdoor scene diversity.
- Day and night scene diversity.
- Relationship between scene complexity and caption length.

**Key result:** The average caption length is approximately **10.49 words**.

The test set contains 100 images and 500 reference captions.

## 10. Dataset Source

**MS COCO 2017**

Official website: https://cocodataset.org/

Images and annotations are downloaded from the official COCO servers and are not redistributed in this repository.

---

_Academic Mini-Project – Image Processing_  
_Comparative Study of Retrieval-Augmented Generation Techniques for Image Captioning._
