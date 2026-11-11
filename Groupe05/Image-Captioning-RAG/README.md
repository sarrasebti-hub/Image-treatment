# Image Captioning Using Retrieval-Augmented Generation (RAG)

## 1. Project Overview

This project investigates the use of Retrieval-Augmented Generation (RAG) techniques for automatic image captioning.

The objective is to develop an image-captioning system that uses visual information and retrieved evidence from a knowledge base to generate relevant, accurate, and visually grounded descriptions of images.

The project uses a subset of the **Microsoft COCO 2017 (MS COCO)** dataset and establishes a shared, reproducible data-processing and knowledge-base foundation for subsequent RAG experiments.

The project is organized into several phases:

- Dataset exploration and statistical analysis.
- Data preprocessing and metadata extraction.
- Image and text embedding generation.
- Knowledge-base construction.
- Quality control and retrieval compatibility validation.
- Validation and freezing of the shared release.
- Implementation and comparison of different RAG techniques.
- Caption generation, evaluation, and statistical analysis.

## 2. Project Objectives

The main objectives are:

1. Explore and analyse the MS COCO dataset.
2. Establish reproducible training, validation, and test splits.
3. Apply consistent image and caption preprocessing.
4. Generate image and text embeddings using selected encoders.
5. Construct a vector-based knowledge base.
6. Validate data integrity, embedding quality, document relationships, and leakage prevention.
7. Prepare a shared and reproducible dataset release for subsequent experiments.
8. Compare different RAG approaches for image captioning..

## 3. Dataset

### 3.1 Dataset Source

The project uses the MS COCO 2017 dataset:

- Training images: `train2017`
- Validation images: `val2017`
- Captions: COCO caption annotations
- Object annotations: COCO instance annotations

The original dataset is stored separately from the processed artifacts.

### 3.2 Dataset Splits

The repository currently contains a generated release using the default split:

| Split | Source | Images | Captions |
|---|---|---:|---:|
| Training / Knowledge Base | COCO train2017 | 9,900 | 49,500 |
| Validation / Tuning | COCO train2017 | 100 | 500 |
| Test / Evaluation | COCO val2017 | 100 | 500 |
| **Total** | | **10,100** | **50,500** |

The training and validation subsets together represent the selected 10,000 images from COCO train2017.

The test subset is drawn from COCO val2017 and remains separate from the training knowledge base.

The split is configuration-driven. To create a different release, update
`01_src/config/pipeline_config.py`, then regenerate the split-dependent
artifacts. Do not mix artifacts generated from different configurations.

To regenerate the complete Phase 1 release with one command, run this from
the project root:

```text
python 01_src/run_pipeline.py
```

The runner generates splits, extracts annotations and images, validates the
dataset, executes preprocessing, regenerates embeddings and knowledge-base
artifacts, and runs quality-control and retrieval validation. It executes
notebooks in a temporary output directory so source notebooks are not
overwritten. Use `--skip-notebooks` for the split, extraction, and validation
stages only.

### 3.3 Data Leakage Prevention

The following constraints are applied:

- Training images and captions form the knowledge base.
- Validation images are reserved for validation and tuning.
- Test images and their reference captions are excluded from the knowledge base.
- Validation and test references must not be used as retrieved evidence for their corresponding evaluation images.
- Split IDs are stored in versioned JSON files.

The split seed is `45`.

## 4. Repository Structure

```text
Image-Captioning-RAG/
│
├── 00_original_data/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│
├── 01_src/
│   ├── data/
│   │   ├── split_dataset.py
│   │   ├── extract_annotations.py
│   │   ├── validate_dataset.py
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       ├── image_preprocessing.py
│   │       └── text_preprocessing.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── clip_image_embeddings.py
│   │   ├── sbert_caption_embeddings.py
│   │   └── embedding_utils.py
│   │
│   └── notebooks/
│       ├── 00_dataset_inspection.ipynb
│       ├── 01_statistical_exploration.ipynb
│       ├── 02_preprocessing_experiments.ipynb
│       ├── 03_embedding_experiments.ipynb
│       ├── 04_knowledge_base_quality_control.ipynb
│       └── 05_retrieval_pipeline_validation.ipynb
│
├── 02_processed_data/
│   ├── training/
│   ├── validation/
│   ├── test/
│   ├── splits/
│   ├── embedding_experiments/
│   ├── preprocessing_validation.json
│   └── knowledge_base/
│       ├── indexes/
│       │   ├── clip_image_index/
│       │   └── sbert_caption_index/
│       ├── documents/
│       │   ├── image_documents.json
│       │   └── caption_documents.json
│       ├── metadata/
│       │   ├── knowledge_base_metadata.json
│       │   ├── quality_control_report.json
│       │   └── quality_control_report.md
│       └── manifest/
│           └── knowledge_base_manifest.json
│
├── 03_analysis/
│   └── 01_statistical_exploration/
│       ├── outputs/
│       │   ├── tables/
│       │   ├── figures/
│       │   └── statistics/
│       └── report/
│           └── statistical_exploration.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

*Note: The structure documents the current organization and the intended location of the project artifacts. Update it if additional files or directories are introduced.*

## 5. Development Environment

The project was developed and tested using:

| Component | Version / Configuration |
|---|---|
| Operating system | Windows 11 |
| Python | 3.11.4 |
| PyTorch | 2.15.0+cpu |
| Transformers | 5.22.1 |
| Pillow | 10.3.0 |
| Sentence-Transformers | 6.2.0 |
| FAISS | 1.15.1 |
| Execution environment | Jupyter Notebook |
| Hardware | CPU |

### 5.1 Models

**Image encoder**

- Model: `openai/clip-vit-base-patch32`
- Embedding dimension: 512
- Output: normalized image embeddings.

**Text encoder**

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Output: normalized caption embeddings.

The CLIP and Sentence-BERT embeddings belong to different embedding spaces. They are stored and indexed separately and must not be compared directly.

## 6. Phase 1 — Dataset Exploration and Preprocessing

### 6.1 Dataset Inspection

The dataset inspection phase verified:

- Availability of the COCO images and annotation files.
- Image readability.
- Caption and annotation availability.
- Dataset split organization.
- Configuration-driven split generation and validation.
- Image-caption relationships.

### 6.2 Statistical Exploration

Statistical exploration was conducted on the selected 9,900 training images.

The analysis covered:

- Object-category frequency.
- Scene complexity.
- Caption length distribution.
- Vocabulary size.
- Part-of-speech distribution.
- Caption-derived attribute mentions.
- Object-category co-occurrence.
- Statistical tables and visualizations.

Some recorded findings:

| Statistic | Result |
|---|---:|
| Training images | 9,900 |
| Training captions | 49,500 |
| Object annotations | 71,326 |
| Object categories | 80 |
| Average objects per image | 7.20 |
| Median objects per image | 4 |
| Average caption length | 10.462 words |
| Vocabulary size | 9,514 |

The statistical exploration report and its associated tables and figures are stored in:

`03_analysis/01_statistical_exploration/`

### 6.3 Image Preprocessing

The image preprocessing pipeline includes:

- EXIF orientation correction.
- RGB conversion.
- Image loading and readability checks.
- CLIP-compatible image processing.

The official CLIP processor performs the model-specific resizing and cropping operations required by the encoder.

### 6.4 Caption Preprocessing

Caption preprocessing uses conservative whitespace normalization.

Original caption meaning, capitalization, punctuation, word order, and stopwords are preserved.

Both original and cleaned caption data are retained to distinguish reference text from retrieval/indexing text.

### 6.5 Preprocessing Validation

The preprocessing validation checks confirmed:

- 9,900 readable training images.
- 100 readable validation images.
- 100 readable test images.
- No detected corrupt images.
- Correct caption counts.
- No empty cleaned captions.
- Correct CLIP input tensor dimensions.

The validation summary is stored in:

`02_processed_data/preprocessing_validation.json`

## 7. Embedding Generation

### 7.1 CLIP Image Embeddings

CLIP was used to generate image embeddings for all three splits.

| Split | Images | Dimensions |
|---|---:|---:|
| Training | 9,900 | 512 |
| Validation | 100 | 512 |
| Test | 100 | 512 |

Embedding properties:

- Model: `openai/clip-vit-base-patch32`
- Data type: float32
- Normalization: L2
- Device: CPU
- Training batch size: 16

The embeddings are saved as NumPy arrays (`.npy`) with corresponding JSON row mappings.

### 7.2 Sentence-BERT Caption Embeddings

Sentence-BERT was used to encode the 49,500 cleaned training captions.

Embedding properties:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Data type: float32
- Normalization: L2
- Device: CPU
- Batch size: 64

Only training captions are embedded for the current knowledge-base construction.

Validation and test reference-caption embeddings have not been generated.

### 7.3 Embedding Integrity

The embedding validation confirmed:

- Correct embedding dimensions.
- Correct vector counts.
- No NaN or infinite values.
- No zero vectors.
- L2-normalized vectors.
- Correct mapping between vector rows and image/caption IDs.

Embedding metadata is stored in the respective split embedding directories.

## 8. Knowledge-Base Construction

### 8.1 Architecture

The knowledge base contains two separate FAISS indexes.

| Index | Encoder | Vectors | Dimensions |
|---|---|---:|---:|
| CLIP image index | CLIP | 9,900 | 512 |
| SBERT caption index | Sentence-BERT | 49,500 | 384 |

### 8.2 Index Configuration

Both indexes use:

- FAISS `IndexFlatIP`.
- Inner-product similarity.
- L2-normalized embeddings.

Since the vectors are L2-normalized, inner product is equivalent to cosine similarity.

### 8.3 Document Structure

Two document collections preserve the relationship between embeddings and their source data.

**Image documents**

Each image document contains:

- Image ID.
- Filename.
- Embedding model and index information.
- Embedding row.
- Associated caption IDs.
- COCO metadata.
- Image metadata.

**Caption documents**

Each caption document contains:

- Caption ID.
- Associated image ID.
- Filename.
- Caption text.
- Embedding model and index information.
- Embedding row.

### 8.4 Knowledge-Base Statistics

| Artifact | Records / Vectors |
|---|---:|
| CLIP image index | 9,900 |
| SBERT caption index | 49,500 |
| Image documents | 9,900 |
| Caption documents | 49,500 |
| **Total indexed vectors** | **59,400** |

The original NumPy embedding matrices are preserved separately from the FAISS indexes.

## 9. Quality Control

Quality control was conducted using:

`01_src/notebooks/04_knowledge_base_quality_control.ipynb`

The following checks passed:

1. Embedding integrity.
2. Mapping uniqueness and row alignment.
3. Document-to-embedding consistency.
4. Image-caption relationship integrity.
5. Dataset leakage audit.
6. FAISS index integrity.
7. Metadata and manifest consistency.
8. Statistical output verification.
9. Retrieval compatibility validation.

The QC reports are stored in:

- `02_processed_data/knowledge_base/metadata/quality_control_report.json`
- `02_processed_data/knowledge_base/metadata/quality_control_report.md`

### 9.1 Retrieval Compatibility Smoke Test

A minimal retrieval compatibility test was conducted using a validation image.

Test configuration:

| Property | Value |
|---|---|
| Query image ID | 250 |
| Query split | Validation |
| CLIP model | openai/clip-vit-base-patch32 |
| Query vector dimension | 512 |
| FAISS index | IndexFlatIP |
| Retrieved images | 3 |

Retrieved training images:

| Rank | Image ID | Similarity score |
|---:|---:|---:|
| 1 | 11605 | 0.8479 |
| 2 | 401670 | 0.8389 |
| 3 | 486769 | 0.8303 |

All 15 associated captions were successfully loaded and verified against their corresponding image documents.

This test confirms compatibility between the frozen CLIP encoder, FAISS index, image documents, and caption documents.

It is a technical smoke test, not a retrieval-quality evaluation or a comparison of RAG techniques.

## 10. Validation and Freezing — Pending

The formal validation and freezing stage remains to be completed.

The project specification requires the 100-image validation subset to be used to select the best preprocessing configuration. The selected preprocessing output must then be frozen and shared with all groups.

Planned tasks:

1. Review the current preprocessing configuration.
2. Validate the selected configuration using the validation subset.
3. Document the selected preprocessing settings and justification.
4. Verify the completeness of the release artifacts.
5. Freeze the approved dataset and knowledge base.
6. Record release versions and checksums.
7. Prepare the shared release for all groups.

The test split and its reference captions must remain protected throughout subsequent experiments.

## 11. Reproducibility

The project uses notebooks for sequential development and Python modules for reusable data-processing and embedding operations.

Recommended execution order:

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `00_dataset_inspection.ipynb` | Inspect dataset and annotations |
| 2 | `01_statistical_exploration.ipynb` | Explore dataset statistics |
| 3 | `02_preprocessing_experiments.ipynb` | Preprocess images and captions |
| 4 | `03_embedding_experiments.ipynb` | Generate and validate embeddings |
| 5 | `04_knowledge_base_quality_control.ipynb` | Construct/inspect and validate KB artifacts |
| 6 | `05_retrieval_pipeline_validation.ipynb` | Validate frozen-release compatibility |
| 7 | `06_validation_and_freezing.ipynb` | Formal validation and release freezing |

Previously generated artifacts should not be rebuilt unnecessarily. Use the saved metadata and manifests to verify existing outputs.

## 12. Data and Version Control Policy

GitHub is used to manage source code, notebooks, documentation, and project reports.

The original COCO dataset and large processed data artifacts should not be committed to the repository.

Recommended version-controlled content:

- Python source files.
- Jupyter notebooks.
- README and technical documentation.
- Requirements file.
- Statistical reports and selected visualizations.
- Small configuration files and metadata where appropriate.

Large dataset files, embeddings, and FAISS indexes should be stored in the designated shared data storage.

## 13. Current Project Status

| Component | Status |
|---|---|
| Dataset inspection | Completed |
| Dataset splitting | Completed |
| Statistical exploration | Completed |
| Image preprocessing | Completed |
| Caption preprocessing | Completed |
| Metadata extraction | Completed |
| CLIP image embeddings | Completed |
| Sentence-BERT caption embeddings | Completed |
| FAISS knowledge-base construction | Completed |
| Document construction | Completed |
| Quality-control checks | Passed |
| Retrieval compatibility smoke test | Passed |
| Formal validation | Pending |
| Final release freezing | Pending |
| RAG technique implementation | Not started |
| Caption generation and evaluation | Not started |

## 14. Future Work

Following validation and freezing, subsequent project phases will focus on:

- Implementation of the assigned RAG technique.
- Integration of a multimodal caption-generation model.
- Hyperparameter tuning using validation data.
- Baseline and RAG comparisons.
- Caption-quality evaluation using BLEU, METEOR, ROUGE-L, CIDEr, and SPICE.
- Retrieval evaluation.
- Hallucination analysis.
- Statistical testing.
- Ablation studies.
- Qualitative analysis and reporting.

## 15. References

- Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision.
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Johnson, J., Douze, M. and Jégou, H. (2019). Billion-scale similarity search with GPUs.

---

**Project status:** Phase 1 implementation and quality control completed. Formal validation and freezing remain the next tasks.
