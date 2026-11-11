# COCO RAG Captioning - Group 28 (Metadata-Filtered RAG)

## Project Overview

This project investigates how different Retrieval-Augmented Generation (RAG) techniques influence the quality, relevance, and visual grounding of automatically generated image captions.

**Assigned Technique:** Metadata-Filtered RAG
**Dataset:** MS COCO 2017 (10,000 training images, 100 validation images, 1,000 test images)

## Phase 1: Data Exploration and Preprocessing

Currently, the project is in **Phase 1**. We have successfully:

1. Extracted and structured the COCO dataset (Train: 10k, Val: 100, Test: 1k).
2. Cleaned and preprocessed the images and captions.
3. Generated statistical analyses and visualizations (object distributions, scene complexity, POS distributions).
4. Frozen the dataset splits to ensure no data leakage between train, validation, and test sets.

## Repository Structure

```text
coco-rag-g28/
│
├── data/                      # (Ignored in Git) Local storage for datasets
│   ├── annotations/           # COCO JSON files (captions, instances, etc.)
│   ├── train/                 # Train split (images, ids.json, captions.json)
│   ├── validation/            # Validation split (images, ids.json)
│   ├── test/                  # Test split (1,000 images, ids.json, metadata)
│   ├── embeddings/            # Generated image/text embeddings
│   └── statistical_analysis/  # Generated CSVs and PNG plots for the report
│
├── notebooks/                 # Jupyter/Colab Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_analysis.ipynb
│   ├── 04_data_splitting_metadata.ipynb
│   ├── 05_embedding_preparation.ipynb
│   └── 06_Testing_Dataset_Preparation.ipynb
│visualisation/
├── src/                       # Python source code/modules
├── figures/                   # Final plots and visualizations for the report
├── .gitignore                 # Excludes heavy datasets from GitHub
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

```
