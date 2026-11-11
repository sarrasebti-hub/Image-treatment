# Image Processing Project — RAG for Image CaptioninG

## Description
Comparative study of RAG (Retrieval-Augmented Generation) techniques for image caption generation on a subset of MS COCO.

## Team
- **Member 1:** Data Engineer (Preprocessing)
- **Member 2:** RAG Engineer (FAISS + Cross-Encoder Reranking)
- **Member 3:** Generation Engineer (BLIP-2)
- **Member 4:** Evaluation Engineer (Metrics + Statistical Tests)

## Project Structure
```text
ImageCaptioning-RagReranking/
├── data/                  # Raw and processed data
├── notebooks/             # Jupyter notebooks
├── outputs/               # FAISS index, embeddings, splits
├── results/               # Final figures, tables, CSV
├── src/                   # Python source code
│   ├── retrieval/         # FAISS building scripts
│   ├── reranking/         # Cross-Encoder reranker
│   ├── captioning/        # BLIP-2 generation pipeline
│   └── evaluation/        # Metrics and statistical tests
└── reports/               # Final report (15-20 pages) 
```
## Installation
```
pip install -r requirements.txt
```

## Usage
To reproduce the results:
Run src/retrieval/build_faiss.py to build the search engine.
Run notebooks/03_generation.ipynb to generate captions.
Run src/evaluation/ scripts to compute metrics.

## Dataset
MS COCO 2017:
10,000 KB images (train2017)
100 validation images (train2017, separate from KB)
100 / 1,000 test images (val2017)

## Method
Visual analysis: CLIP image embeddings (512-dim)
Retrieval: FAISS Top-20 similar images
Reranking: Cross-Encoder to keep Top-5
Generation: BLIP-2 caption with retrieved examples
Evaluation: CIDEr, BLEU-4, METEOR, SPICE + statistical tests

## Results
Baseline vs RAG comparison on test images.
Primary metric: CIDEr.
Statistical test: Wilcoxon signed-rank (α = 0.05).

## Group
Group 11 — RAG Reranking