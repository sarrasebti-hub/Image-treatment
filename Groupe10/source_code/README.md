# RAG-Based Image Captioning

## Project Overview

This project explores a Retrieval-Augmented Generation (RAG) approach for image captioning using the MS COCO 2017 dataset.

The system combines:
- **CLIP** for image and text feature extraction.
- **YOLOv8** for object detection and metadata extraction.
- **FAISS** for similarity search and vector storage.
- **COCO annotations** for reference captions and object labels.

The objective is to prepare a structured image-caption dataset, extract visual and textual features, and build a retrieval system that can support a future image captioning model.

## Dataset

The project uses the MS COCO 2017 dataset with three separate splits:

| Split | Images | Purpose |
|---|---:|---|
| Knowledge Base | 10,000 | Retrieval database |
| Validation | 100 | Preprocessing validation |
| Test | 1,000 | Held-out evaluation |

The splits are designed to avoid image ID overlap.

Reference captions are retained from the original COCO annotations. The number of captions may vary slightly between images because the original dataset does not contain exactly five captions for every image.

## Technologies

- Python
- NumPy
- Pillow
- PyTorch
- Hugging Face Transformers
- CLIP (`openai/clip-vit-base-patch32`)
- Ultralytics YOLOv8
- FAISS

## Project Structure

```text
RAG-Based-Image-Captioning/
│
├── data/
│   ├── raw/
│   │   └── annotations/
│   └── processed/
│       ├── knowledge_base/
│       ├── validation/
│       ├── test/
│       ├── splits/
│       └── yolo/
│
├── src/
│   ├── analysis/
│   │   ├── analyze_categories.py
│   │   └── analyze_detected_objects.py
│   │
│   ├── audit/
│   │   ├── check_images.py
│   │   ├── check_split_duplicates.py
│   │   ├── cleanup_raw_images.py
│   │   ├── create_splits.py
│   │   ├── prepare_dataset.py
│   │   └── verify_dataset.py
│   │
│   ├── embeddings/
│   │   ├── build_embeddings.py
│   │   ├── build_vector_db.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── encoder.py
│   │   ├── pipeline.py
│   │   ├── quality_control.py
│   │   └── search.py
│   │
│   ├── preprocessing/
│   │   ├── expand_test_split.py
│   │   ├── extract_metadata.py
│   │   ├── extract_objects.py
│   │   └── preprocess_images.py
│   │
│   └── training/
│       ├── create_data_yaml.py
│       ├── prepare_validation_labels.py
│       ├── prepare_yolo_dataset.py
│       ├── train_yolo.py
│       └── validate_yolo_labels.py
│
├── artifacts/
│   ├── embeddings/
│   ├── metadata/
│   └── vector_db/
│
├── reports/
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shahinazchrf/RAG-Based-Image-Captioning.git
cd RAG-Based-Image-Captioning
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The project requires the COCO images and annotation files in the expected `data/` directories before running the complete pipeline.

## Processing Pipeline

The project is organized into the following stages:

1. **Dataset Audit:** Verify image integrity, annotations, split membership, and missing files.
2. **Dataset Preparation:** Create the Knowledge Base, Validation, and Test splits.
3. **Image Preprocessing:** Verify image readability and convert images to RGB during encoding.
4. **Object Detection:** Use YOLOv8 to extract object labels, confidence scores, and bounding boxes.
5. **Embedding Extraction:** Encode images and captions using CLIP.
6. **Vector Database Construction:** Build FAISS indices from Knowledge Base embeddings.
7. **Quality Control:** Check embedding validity, metadata consistency, retrieval behavior, and duplicate vectors.

## Running the Embedding Pipeline

To display available options:

```bash
python -m src.embeddings.pipeline --help
```

To run the complete embedding and vector database pipeline using all Knowledge Base images and the Test split:

```bash
python -m src.embeddings.pipeline --kb-limit 0 --eval-split test
```

The pipeline supports additional options such as batch size, model selection, index type, evaluation limits, and quality-control settings.

## Embeddings and Retrieval

The project uses the CLIP model:

`openai/clip-vit-base-patch32`

Image and text embeddings share a 512-dimensional feature space.

The encoder:
- Converts images to RGB.
- Uses the CLIP processor for image resizing, cropping, and pixel normalization.
- Tokenizes and truncates text to a maximum length of 77 tokens.
- Applies L2 normalization to embedding vectors.

FAISS uses inner-product similarity on normalized vectors, which corresponds to cosine similarity.

The vector database contains three indices:

- `image_index.faiss`
- `caption_index.faiss`
- `joint_index.faiss`

The indices are constructed using Knowledge Base embeddings only.

## Quality Control

Quality-control checks include:
- Embedding dimensions and data types.
- Finite values and zero vectors.
- L2 normalization.
- Metadata and image-caption consistency.
- Duplicate and near-duplicate embeddings.
- Image-to-text and text-to-image retrieval metrics.

The current retrieval evaluation uses a limited set of 200 queries. These results are baseline measurements and should not be interpreted as final model performance.

## Current Phase 1 Outputs

Generated files include:

- Processed dataset splits and captions.
- Image metadata and detected-object metadata.
- CLIP image and text embeddings.
- FAISS vector indices.
- Dataset audit and preprocessing reports.
- Object analysis and visualization reports.
- Quality-control metrics.

The generated artifacts are stored in `artifacts/`, while analysis reports and visualizations are stored in `reports/`.

## Important Notes

- Validation and Test images are kept separate from the Knowledge Base.
- Test data must not be used for model training or preprocessing selection.
- Original COCO reference captions should remain unchanged for evaluation.
- YOLO smoke-training results are pipeline checks, not final model results.
- The current CLIP encoder uses pretrained weights and has not been fine-tuned for this project.

## Project Status

Phase 1 focuses on dataset exploration, preprocessing, metadata extraction, embedding generation, vector database construction, and quality control.

Further model development and evaluation are planned for subsequent phases.
