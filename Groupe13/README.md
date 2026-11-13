# Group 13 – Multimodal RAG for Image Caption Generation

## Project Title

**A Comparative Study of RAG Techniques for Generating Image Captions – Multimodal RAG**

## Group Members

- SOUCHANE Serine
- SEBA Bouchra
- MERADI Serine Feriel
- AGGOUNE Abderrahim

---

## Project Overview

This project explores the use of Retrieval-Augmented Generation (RAG) techniques for image caption generation.

The objective is to build a multimodal system that uses visual and textual information to retrieve relevant knowledge and support image caption generation.

The project uses the **Microsoft COCO 2017 dataset**, which provides images, captions, and object annotations.

## Phase 1 – Dataset Preparation and Knowledge Base Construction

Phase 1 focuses on preparing, analyzing, and organizing the dataset to build the knowledge base required for the multimodal RAG system.

The main tasks include:

1. **Dataset audit:** checking the dataset structure, image availability, and data consistency.
2. **Dataset splitting:** creating separate knowledge base, validation, and test partitions.
3. **Caption preprocessing:** extracting, cleaning, and organizing image captions.
4. **Metadata extraction:** collecting object information and spatial relationships.
5. **Visual detection:** performing supplementary object-detection experiments using YOLO.
6. **Exploratory data analysis:** generating statistics and visualizations to understand the dataset.
7. **Embedding generation:** generating image and text embeddings using CLIP.
8. **Quality control:** checking embedding dimensions, missing values, and dataset consistency.
9. **Retrieval validation:** conducting preliminary image-to-image, image-to-text, text-to-image, and text-to-text retrieval experiments.

## Technologies and Tools

- Python
- Microsoft COCO 2017
- CLIP (`openai/clip-vit-base-patch32`)
- YOLO
- NumPy
- FAISS
- Matplotlib
- Pandas
- spaCy

## Repository Structure

```text
Groupe13/
│
├── README.md
├── DATASET.md
├── requirements.txt
│
├── data/
│   ├── captions/
│   ├── embeddings/
│   ├── exploration/
│   ├── index/
│   ├── metadata/
│   ├── splits/
│   ├── visual_detections/
│   └── visual_detections_test/
│
├── docs/
│   ├── final_report.pdf
│   └── latex/
│
├── figures/
│
├── reports/
│
└── scripts/
```

## Documentation and Results

- `DATASET.md`: dataset description, partitions, preprocessing, and data organization.
- `docs/final_report.pdf`: detailed project report.
- `reports/`: audit, preprocessing, quality-control, and retrieval-validation reports.
- `figures/`: generated graphs and visualizations.
- `scripts/`: Python scripts used throughout Phase 1.

## Dataset Availability

The original COCO images and large generated artifacts are not included in this repository because of their size.

Refer to `DATASET.md` for dataset details, partition information, and instructions concerning the required data.

## Project Status

**Phase 1 – Dataset preparation, analysis, and knowledge base construction.**

Further RAG experiments and comparisons are part of the subsequent project work.
