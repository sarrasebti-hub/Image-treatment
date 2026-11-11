# Pipeline Workflow and Reproducibility Guide

## 1. Purpose

This document describes how to execute and reproduce the Image Captioning RAG data pipeline.

It provides a standard workflow for:

- Preparing the MS COCO dataset.
- Generating training, validation, and test splits.
- Extracting and validating image and caption data.
- Exploring dataset statistics.
- Applying image and text preprocessing.
- Generating image and caption embeddings.
- Building the vector indexes and knowledge-base documents.
- Performing quality-control checks.
- Validating and freezing a dataset release.

The pipeline is designed to be reusable. Important configuration parameters, such as the random seed and split sizes, should be changeable without manually modifying multiple notebooks or scripts.

**Important:** Changing a configuration parameter may require regenerating several downstream artifacts. The pipeline must therefore be executed in the correct order.

---

## 2. Pipeline Overview

The complete workflow is organized into the following stages:

```text
Original MS COCO Dataset
          |
          v
1. Dataset Configuration
          |
          v
2. Dataset Splitting
          |
          v
3. Annotation Extraction
          |
          v
4. Dataset Validation
          |
          v
5. Statistical Exploration
          |
          v
6. Image and Text Preprocessing
          |
          v
7. Embedding Generation
          |
          v
8. Knowledge-Base Construction
          |
          v
9. Quality Control
          |
          v
10. Retrieval Compatibility Validation
          |
          v
11. Preprocessing Validation
          |
          v
12. Dataset Release and Freezing
```

Each stage depends on the outputs of previous stages. If an earlier stage changes, its dependent stages must be reconsidered or regenerated.

---

## 3. Project Configuration

### 3.1 Centralized configuration

All parameters that may change between pipeline executions should be defined in one configuration file.

Recommended location:

`01_src/config/pipeline_config.py`

Example configuration:

```python
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_ROOT = PROJECT_ROOT / "00_original_data"

PROCESSED_DATA_ROOT = PROJECT_ROOT / "02_processed_data"

ANALYSIS_ROOT = PROJECT_ROOT / "03_analysis"


# Reproducibility
RANDOM_SEED = 45


# Dataset configuration
TRAIN_SOURCE = "train2017"
TEST_SOURCE = "val2017"

TRAINING_SIZE = 9900
VALIDATION_SIZE = 100
TEST_SIZE = 100

CAPTIONS_PER_IMAGE = 5


# Embedding models
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

SBERT_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Embedding configuration
CLIP_BATCH_SIZE = 32
SBERT_BATCH_SIZE = 64

EMBEDDING_DTYPE = "float32"
NORMALIZE_EMBEDDINGS = True


# Retrieval configuration
RETRIEVAL_TOP_K = 3
```

These values are the active centralized configuration used by the splitting,
extraction, validation, preprocessing, embedding, and knowledge-base checks.
The generated `split_config.json` records a snapshot of the configuration used
for each dataset release.

### 3.2 Configuration principles

1. Avoid hardcoding the random seed in multiple scripts.
2. Avoid repeating split sizes in different notebooks.
3. Keep model names and embedding parameters centralized.
4. Record the configuration used for every pipeline execution.
5. Do not overwrite an existing frozen release without explicitly creating a new version.

---

## 4. Execution Workflow

### Stage 1 — Dataset configuration

**Purpose:** Define the dataset sources, split sizes, random seed, and output directories.

Check that the original dataset contains the required folders:

```text
00_original_data/
├── train2017/
├── val2017/
└── annotations/
```

Verify that the required COCO annotation files are available.

Expected configuration output:

- Random seed.
- Training size.
- Validation size.
- Test size.
- Dataset source directories.
- Output directories.

### Stage 2 — Dataset splitting

**Purpose:** Select the image IDs assigned to each dataset split.

Notebook:

`01_src/notebooks/00_dataset_inspection.ipynb`

Script:

`01_src/data/split_dataset.py`

The current configuration is:

| Split | Source | Images | Captions |
|---|---|---:|---:|
| Training / Knowledge Base | train2017 | 9,900 | 49,500 |
| Validation | train2017 | 100 | 500 |
| Test | val2017 | 100 | 500 |
| **Total** | | **10,100** | **50,500** |

The training and validation subsets are selected from the same COCO training source. The test subset is selected independently from COCO validation images.

Use the configured random seed when selecting image IDs.

Expected outputs:

```text
02_processed_data/
└── splits/
    ├── training_ids.json
    ├── validation_ids.json
    ├── test_ids.json
    └── split_config.json
```

Validation requirements:

- The number of IDs in each split must match the configuration.
- Training and validation IDs must not overlap.
- Test IDs must not overlap with training or validation IDs.
- The selected IDs must exist in their respective COCO sources.
- The split configuration must record the seed and split sizes.

### Stage 3 — Annotation extraction

**Purpose:** Extract the captions and object annotations corresponding to the selected image IDs.

Script:

`01_src/data/extract_annotations.py`

Expected outputs include:

- Captions associated with training images.
- Captions associated with validation images.
- Captions associated with test images.
- Object annotations and category information where required.

The extraction process must use the split ID files generated in Stage 2.

Expected caption counts:

- Training: 49,500.
- Validation: 500.
- Test: 500.

### Stage 4 — Dataset validation

**Purpose:** Verify that the extracted dataset is complete and internally consistent.

Script:

`01_src/data/validate_dataset.py`

Checks should include:

- Image files exist.
- Image files can be opened.
- Caption counts match expectations.
- Every caption references an existing image.
- Image IDs and caption IDs are unique where required.
- No image or caption is assigned to an incorrect split.

Do not proceed if these checks fail.

### Stage 5 — Statistical exploration

Notebook:

`01_src/notebooks/01_statistical_exploration.ipynb`

**Purpose:** Understand the dataset before applying preprocessing and building the knowledge base.

Analyze:

- Object category distribution.
- Number of annotated objects per image.
- Caption length distribution.
- Vocabulary size and token frequency.
- Scene complexity.
- Relevant caption characteristics.

Expected outputs:

```text
03_analysis/
└── 01_statistical_exploration/
    ├── outputs/
    │   ├── tables/
    │   ├── figures/
    │   └── statistics/
    └── report/
        └── statistical_exploration.md
```

### Stage 6 — Image and text preprocessing

Notebook:

`01_src/notebooks/02_preprocessing_experiments.ipynb`

Modules:

- `01_src/data/preprocessing/image_preprocessing.py`
- `01_src/data/preprocessing/text_preprocessing.py`

**Image preprocessing**

The current image pipeline includes:

- Loading images.
- Correcting EXIF orientation.
- Converting images to RGB.
- Applying the CLIP processor to prepare model inputs.

**Text preprocessing**

The current text-cleaning procedure removes unnecessary whitespace while preserving the original caption wording, punctuation, capitalization, and meaning.

Keep original captions available. Never replace the original annotation files with cleaned captions.

Expected outputs:

```text
02_processed_data/
├── training/
├── validation/
└── test/
    └── ...
```

Each split should retain its associated image, caption, and preprocessing metadata.

### Stage 7 — Embedding generation

Notebook:

`01_src/notebooks/03_embedding_experiments.ipynb`

**Purpose:** Convert images and captions into numerical vector representations.

Two separate embedding spaces are currently used.

**A. CLIP image embeddings**

- Model: `openai/clip-vit-base-patch32`
- Dimension: 512
- Input: Images.
- Output: Image embeddings.

Expected training shape:

`(9900, 512)`

**B. SBERT caption embeddings**

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Input: Training captions.
- Output: Caption embeddings.

Expected training shape:

`(49500, 384)`

The two embedding spaces must remain separate. CLIP image vectors and SBERT caption vectors must not be compared directly.

Expected outputs include:

- Embedding NumPy arrays.
- Embedding-to-image or embedding-to-caption mappings.
- Embedding metadata.

Validation requirements:

- Correct vector dimensions.
- Correct number of vectors.
- No NaN or infinite values.
- No zero vectors.
- Correct normalization.
- Correct mapping between embedding rows and dataset records.

### Stage 8 — Knowledge-base construction

Notebook:

`01_src/notebooks/03_embedding_experiments.ipynb`

**Purpose:** Build the vector indexes and document collections used by the retrieval pipeline.

The current implementation creates two separate FAISS indexes:

| Index | Vectors | Dimension | Metric |
|---|---:|---:|---|
| CLIP image index | 9,900 | 512 | Inner product |
| SBERT caption index | 49,500 | 384 | Inner product |

The vectors are L2-normalized, making inner-product search equivalent to cosine-similarity search.

Expected knowledge-base structure:

```text
02_processed_data/
└── knowledge_base/
    ├── indexes/
    │   ├── clip_image_index/
    │   └── sbert_caption_index/
    ├── documents/
    │   ├── image_documents.json
    │   └── caption_documents.json
    ├── metadata/
    │   ├── knowledge_base_metadata.json
    │   ├── quality_control_report.json
    │   └── quality_control_report.md
    └── manifest/
        └── knowledge_base_manifest.json
```

Only training data is included in the knowledge base.

Validation and test images and captions must remain excluded.

### Stage 9 — Quality control

Notebook:

`01_src/notebooks/04_knowledge_base_quality_control.ipynb`

**Purpose:** Verify the integrity of the generated knowledge base.

Required checks:

- Embedding integrity.
- Embedding normalization.
- Embedding-to-document mapping.
- Unique image and caption IDs.
- Correct image-caption associations.
- FAISS index dimensions and vector counts.
- Metadata and manifest consistency.
- Training/validation/test leakage audit.

The quality-control report must record the results of these checks.

Do not mark the release as passed if any mandatory check fails.

### Stage 10 — Retrieval compatibility validation

Notebook:

`01_src/notebooks/05_retrieval_pipeline_validation.ipynb`

**Purpose:** Verify that the frozen knowledge-base artifacts can be loaded and used by a minimal retrieval process.

The current smoke test:

1. Loads the CLIP image index.
2. Loads the image document mappings.
3. Selects an image from the validation split.
4. Generates its CLIP image embedding.
5. Searches the training index.
6. Retrieves the associated image documents.
7. Retrieves the captions associated with those images.

This stage verifies technical compatibility. It does not constitute a complete retrieval-quality evaluation or an experiment comparing different RAG techniques.

The query image and its reference captions must remain excluded from the knowledge base.

### Stage 11 — Preprocessing validation and selection

**Purpose:** Use the 100-image validation subset to evaluate preprocessing configurations and select the configuration to retain.

The selected configuration and its output must be documented.

The current one-image retrieval smoke test does not replace this full validation stage.

Validation references must not be exposed to retrieval when evaluating the corresponding validation images.

### Stage 12 — Dataset release and freezing

**Purpose:** Produce a reproducible, versioned dataset and knowledge-base release.

Before freezing:

- Confirm that all split counts are correct.
- Confirm that all mandatory QC checks pass.
- Confirm that validation and test data are excluded from the knowledge base.
- Confirm that metadata and manifests describe the actual artifacts.
- Confirm that the retrieval compatibility test passes.
- Record the selected preprocessing configuration.
- Record the software and model versions.

Once frozen, the release should not be modified in place.

Any later changes to the seed, split sizes, preprocessing, or embedding models should produce a new release.

---

## 5. Changing Pipeline Parameters

### 5.1 Changing the random seed

The random seed controls the selection of image IDs during dataset splitting.

For example:

```python
RANDOM_SEED = 45
```

Changing it to another value may produce different training and validation image IDs.

When the seed changes, regenerate:

1. Dataset splits.
2. Extracted annotations.
3. Dataset validation results.
4. Split-dependent statistical outputs.
5. Preprocessed datasets.
6. Image and caption embeddings.
7. FAISS indexes.
8. Knowledge-base documents.
9. Metadata and manifests.
10. Quality-control reports.
11. Retrieval compatibility results.

The seed must be recorded in `split_config.json` and the release metadata.

### 5.2 Changing the split sizes

For example, changing the training and validation allocation while keeping the same total number of training-source images:

```python
TRAINING_SIZE = 9800
VALIDATION_SIZE = 200
TEST_SIZE = 100
```

The total number of images selected from `train2017` remains 10,000.

Expected caption counts, assuming five captions per image:

| Split | Images | Captions |
|---|---:|---:|
| Training | 9,800 | 49,000 |
| Validation | 200 | 1,000 |
| Test | 100 | 500 |

When split sizes change, regenerate every downstream artifact that depends on the selected IDs.

In particular, the knowledge base must be rebuilt because its training images and captions have changed.

The minimum command sequence for a new release is:

```text
python 01_src/data/split_dataset.py
python 01_src/data/extract_annotations.py
python 01_src/data/validate_dataset.py
```

Then rerun the preprocessing and embedding notebooks, rebuild the FAISS
indexes and knowledge base, and rerun the quality-control and retrieval
validation notebooks. The validator reads the split sizes and
`CAPTIONS_PER_IMAGE` directly from `pipeline_config.py`; it does not contain
release-specific counts.

Alternatively, run the complete pipeline from the project root:

```text
python 01_src/run_pipeline.py
```

This command runs the three data scripts and the canonical preprocessing,
embedding/knowledge-base, quality-control, and retrieval notebooks in order.
It stops at the first failed stage. Use
`python 01_src/run_pipeline.py --skip-notebooks` when only the split,
annotation extraction, and dataset validation stages are required.

### 5.3 Changing the embedding model

Changing CLIP or SBERT models requires regenerating the corresponding embeddings and vector indexes.

For example:

- Changing CLIP requires regenerating image embeddings and rebuilding the CLIP index.
- Changing SBERT requires regenerating caption embeddings and rebuilding the SBERT index.

The document mappings and metadata must be checked against the new vectors.

Never mix embeddings from different models or incompatible embedding configurations in the same index.

### 5.4 Changing preprocessing

A change to image preprocessing may affect CLIP embeddings.

A change to caption preprocessing may affect SBERT embeddings.

Regenerate the affected embeddings and indexes after changing the corresponding preprocessing procedure.

Keep the previous release available for comparison.

---

## 6. Dependency and Regeneration Rules

The following table summarizes which stages must be revisited after a configuration change.

| Change | Stages to regenerate |
|---|---|
| Random seed | 2–12 |
| Training size | 2–12 |
| Validation size | 2–12 |
| Test size | 2–12 |
| Image preprocessing | 6, 7 (CLIP), 8, 9, 10, 11, 12 |
| Caption preprocessing | 6, 7 (SBERT), 8, 9, 12 |
| CLIP model | 7 (CLIP), 8, 9, 10, 12 |
| SBERT model | 7 (SBERT), 8, 9, 12 |
| FAISS index configuration | 8, 9, 10, 12 |

The table assumes that the original dataset and split assignments remain unchanged for preprocessing and model changes.

When a change affects multiple dependencies, regenerate the union of the affected stages.

---

## 7. Reproducibility and Versioning

Every pipeline execution should have a recorded configuration.

Recommended release structure:

```text
02_processed_data/
└── releases/
    ├── release_001/
    ├── release_002/
    └── ...
```

Each release should contain or reference:

- Split configuration.
- Random seed.
- Split ID files.
- Preprocessing configuration.
- Embedding model names and versions.
- Embedding dimensions and normalization settings.
- Knowledge-base metadata.
- Quality-control report.
- Manifest.
- Software environment information.

A release identifier should be unique.

For example:

`release_001_seed45_train9900_val100_test100`

Do not overwrite an existing frozen release. Create a new release when the configuration changes.

---

## 8. Recommended Execution Order

For a fresh pipeline execution, follow this order:

1. Verify the original dataset.
2. Load the pipeline configuration.
3. Generate the dataset splits.
4. Extract captions and object annotations.
5. Validate the extracted dataset.
6. Run statistical exploration.
7. Run image and text preprocessing.
8. Generate CLIP image embeddings.
9. Generate SBERT caption embeddings.
10. Build the FAISS indexes.
11. Build the knowledge-base documents.
12. Run quality-control checks.
13. Run retrieval compatibility validation.
14. Complete preprocessing validation and selection.
15. Generate and freeze the release.

Do not skip the quality-control stage.

---

## 9. Important Data and Leakage Rules

The following rules apply to every pipeline execution:

1. The knowledge base contains training data only.
2. Validation reference captions must not be used to retrieve information for their corresponding validation queries.
3. Test reference captions must never be included in the knowledge base.
4. The test split must not be modified during model or preprocessing selection.
5. Original annotations must be preserved.
6. Every split must have a recorded configuration.
7. Every frozen release must be reproducible from its recorded configuration and source data.

---

## 10. Current Implementation Status

The following table distinguishes implemented work from remaining work.

| Stage | Status |
|---|---|
| Dataset configuration and splitting | Implemented |
| Annotation extraction | Implemented |
| Dataset validation | Implemented |
| Statistical exploration | Completed |
| Image and text preprocessing | Implemented |
| CLIP image embeddings | Completed |
| SBERT caption embeddings | Completed |
| Knowledge-base construction | Completed |
| Knowledge-base quality control | Passed |
| Retrieval compatibility smoke test | Passed |
| Full preprocessing validation and selection | Implemented through `01_src/run_pipeline.py` |
| Final release freezing | Pending |

**Note:** The centralized configuration file and fully automated end-to-end execution are proposed improvements. They should not be considered implemented until the existing scripts and notebooks have been connected to the shared configuration and tested together.

---

## 11. Final Principle

The pipeline should be reproducible, traceable, and safe to rerun.

A change to one parameter must have predictable consequences. The project should never depend on manually remembering which files need to be regenerated.

The configuration, split files, metadata, manifests, and quality-control reports together provide the information required to understand and reproduce each dataset release.