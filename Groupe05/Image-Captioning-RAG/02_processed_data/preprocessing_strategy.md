# Preprocessing Strategy
## Image Captioning RAG Project

**Version:** 1.0  
**Phase:** Phase 1 — Data Preprocessing  
**Image encoder:** CLIP  
**Text encoder:** Sentence-BERT  
**Dataset:** MS-COCO 2017 subset

---

## 1. Purpose

The purpose of preprocessing is to transform the selected MS-COCO dataset into a clean, consistent, reproducible, and structured collection of data suitable for image and text embedding generation and knowledge-base construction.

The preprocessing pipeline will prepare:

- Images for the CLIP image encoder.
- Captions for the Sentence-BERT text encoder.
- Object annotations and associated metadata.
- Quality-control information to verify the integrity of the resulting dataset.

The same general preprocessing configuration will be applied consistently across the training, validation, and test splits, while respecting the different roles and access restrictions of each split.

## 2. Dataset configuration

The project uses the following dataset configuration:

| Split | Source | Images | Captions |
|---|---|---:|---:|
| Training | COCO train2017 | 9,900 | 49,500 |
| Validation | Selected COCO train2017 images | 100 | 500 |
| Testing | Selected COCO val2017 images | 100 | 500 |

Each image has five reference captions.

The training and validation subsets together form the selected 10,000 images from COCO train2017. The test subset is separate and comes from COCO val2017.

## 3. Preprocessing principles

The following principles will guide implementation:

1. **Preservation of original data:** Original images, captions, and annotations must remain unchanged.
2. **Reproducibility:** Preprocessing operations and configurations must be documented and repeatable.
3. **Consistency:** The same encoder-specific image and text processing configurations must be used across all splits.
4. **Separation of responsibilities:** Image processing, caption processing, and metadata preparation will be implemented as separate modules.
5. **Anti-leakage:** Validation and test reference captions must never enter the training knowledge base or retrieval index.
6. **Traceability:** Every processed record must remain associated with its original image ID and, where applicable, caption ID.
7. **Non-destructive processing:** Cleaned captions and transformed representations must be stored separately from original data.

## 4. Image preprocessing

### 4.1 Objective

Prepare images in a consistent format suitable for generating image embeddings using CLIP.

### 4.2 Operations

The image preprocessing module will perform the following operations:

1. Verify that each image exists and can be opened.
2. Check image integrity and detect unreadable files.
3. Correct image orientation using EXIF information when necessary.
4. Convert images to RGB.
5. Apply the resizing, cropping, and normalization operations required by the selected CLIP model.
6. Verify the resulting image representation before embedding generation.

### 4.3 CLIP-specific processing

CLIP requires images to be transformed according to the configuration associated with its selected checkpoint.

The Hugging Face image processor will be used to apply the model-specific operations, including:

- Image resizing.
- Cropping, where required.
- RGB conversion.
- Pixel-value rescaling.
- Normalization using the model's configured mean and standard deviation.

The implementation must use the processor associated with the exact CLIP checkpoint rather than manually assuming image dimensions or normalization values.

### 4.4 Preservation

Original images will remain unchanged.

Encoder-ready image representations will be generated from the originals during processing. We will avoid creating unnecessary duplicate image collections unless persistent processed images are required by the implementation.

### 4.5 Expected output

For each image:

- Original image ID.
- Original image path.
- Image dimensions.
- Image format.
- Processing status.
- CLIP-ready image representation when required for embedding generation.

## 5. Caption preprocessing

### 5.1 Objective

Prepare human-written captions for text embedding generation using Sentence-BERT while preserving their original content.

### 5.2 Original caption preservation

The original reference captions must remain unchanged.

Each caption will retain:

- Image ID.
- Caption ID.
- Original caption text.

The original captions will be preserved for evaluation and traceability.

### 5.3 Text normalization

A separate cleaned-text field will be generated for indexing and embedding purposes.

The initial cleaning operations will be conservative:

1. Normalize repeated whitespace.
2. Remove unnecessary leading and trailing whitespace.
3. Preserve meaningful punctuation.
4. Preserve descriptive words and expressions.
5. Avoid unnecessary removal of stopwords.
6. Avoid aggressive stemming or lemmatization.
7. Preserve the original caption separately.

We will not remove information that may be important for describing objects, attributes, actions, or context.

### 5.4 Sentence-BERT-specific processing

Sentence-BERT will perform the model-specific tokenization and input preparation.

The implementation will use the tokenizer associated with the selected Sentence-BERT checkpoint.

The following configuration details will be recorded:

- Model identifier and version.
- Tokenizer configuration.
- Maximum sequence length.
- Truncation settings.
- Embedding dimension.
- Whether embedding normalization is enabled.

The original caption and the cleaned caption will remain available so that preprocessing decisions can be reviewed.

### 5.5 Expected output

For each caption:

- Caption ID.
- Image ID.
- Original caption.
- Cleaned caption.
- Processing status.

The generated text embedding will later be associated with the same caption and image IDs.

## 6. Metadata preparation

### 6.1 Objective

Prepare the object annotations and available image metadata in a consistent structure.

### 6.2 Operations

The metadata preprocessing module will:

1. Read the COCO instance annotations.
2. Map category IDs to category names.
3. Associate object annotations with their corresponding image IDs.
4. Preserve bounding boxes, segmentation information, area, and crowd indicators.
5. Organize annotations by image.
6. Check for invalid category IDs and missing image references.

### 6.3 Metadata limitations

Only metadata available in the source annotations will be treated as ground truth.

Semantic relationships, visual attributes, and scene context will not be invented or assumed to exist in the COCO instance annotations.

Additional visual metadata may be generated in a later stage if the project uses an appropriate detector or model.

## 7. Split-specific processing and access rules

All three splits will undergo the applicable preprocessing operations. However, their data will be used differently.

### 7.1 Training set

The training set is the only source of retrieval evidence for the knowledge base.

It will undergo:

- Image preprocessing for CLIP.
- Caption cleaning for Sentence-BERT.
- Metadata preparation.
- Image embedding generation.
- Text embedding generation.
- Knowledge-base construction.

The 9,900 training images and their 49,500 captions may be indexed according to the selected RAG architecture.

### 7.2 Validation set

The validation set will undergo the same applicable image and text preparation.

Its purpose is to support configuration selection and hyperparameter tuning.

Validation images may be processed and embedded as query inputs. However:

- Validation images must not be inserted into the training knowledge base.
- Validation reference captions must not be indexed.
- An image must not retrieve its own reference captions.
- Reference captions may be used for evaluation after generation.

### 7.3 Test set

The test set will undergo the same applicable image preprocessing configuration.

Its purpose is final evaluation after the system configuration has been selected and frozen.

The following restrictions apply:

- Test images must not be inserted into the training knowledge base.
- Test reference captions must never be indexed.
- Test reference captions must not be used for retrieval or generation.
- Test references are used only for final evaluation.
- Test results must not be used to tune the system.

## 8. Embedding generation considerations

Image and text embeddings will be generated using separate models.

| Representation | Model | Input | Output |
|---|---|---|---|
| Image embedding | CLIP | Processed image | Image vector |
| Text embedding | Sentence-BERT | Cleaned caption | Text vector |

The following information will be recorded for reproducibility:

- Model name and exact checkpoint.
- Model version or revision.
- Embedding dimensions.
- Batch size.
- Numerical precision.
- Embedding normalization configuration.
- Processing time.
- Number of successfully generated embeddings.

### Important compatibility consideration

CLIP image embeddings and Sentence-BERT text embeddings are generated by different models and belong to different vector spaces.

Their raw cosine similarities must not be treated as directly comparable cross-modal similarity scores.

The project will maintain separate image and text embedding collections initially. Any later combination of retrieval scores will require an explicitly defined and evaluated fusion strategy.

## 9. Proposed output organisation

The processed dataset will remain separated by split.

```text
02_processed_data/
│
├── training/
│   ├── images/
│   ├── annotations/
│   ├── captions/
│   └── metadata/
│
├── validation/
│   ├── images/
│   ├── annotations/
│   ├── captions/
│   └── metadata/
│
├── test/
│   ├── images/
│   ├── annotations/
│   ├── captions/
│   └── metadata/
│
└── splits/
    ├── training_ids.json
    ├── validation_ids.json
    ├── test_ids.json
    └── split_config.json
```

The precise files and formats will be finalized during implementation. Original dataset files will remain in `00_original_data/`.

Embeddings and retrieval indexes may be organised as separate artifacts in the subsequent embedding-generation and knowledge-base stages.

## 10. Quality control

The preprocessing pipeline must include validation checks.

### Image validation

- Verify image existence and readability.
- Detect invalid or corrupted images.
- Verify RGB conversion.
- Check image dimensions and output format.
- Verify compatibility with the CLIP image processor.

### Caption validation

- Verify caption-to-image associations.
- Check that original captions remain unchanged.
- Verify that cleaned captions are not empty.
- Check for missing or duplicated caption IDs.
- Verify five reference captions per image.

### Metadata validation

- Check category ID consistency.
- Verify annotation-to-image associations.
- Check bounding-box validity.
- Detect missing metadata records.

### Anti-leakage validation

- Verify that training, validation, and test image IDs are disjoint.
- Verify that only training records enter the knowledge base.
- Verify that validation and test reference captions are excluded from retrieval indexes.
- Verify that an evaluation image cannot retrieve its own reference captions.

### Reproducibility

Each preprocessing run must record:

- Configuration used.
- Model identifiers.
- Processing date.
- Number of processed records.
- Number of failed records.
- Validation results.

## 11. Implementation approach

The preprocessing system will follow a hybrid architecture.

**Python scripts** will contain the reusable preprocessing functions and pipeline.

**Jupyter notebooks** will be used to inspect data, test the modules, compare original and processed samples, and visualise results.

The proposed modules are:

| Module | Responsibility |
|---|---|
| `image_preprocessing.py` | Image loading, validation and CLIP image processing |
| `caption_preprocessing.py` | Caption validation and text cleaning |
| `metadata_preprocessing.py` | Object annotation and metadata preparation |
| `preprocessing_pipeline.py` | Coordination of preprocessing operations |

The notebook `02_preprocessing_experiments.ipynb` will be used to test and verify the implementation.

## 12. Completion criteria

Preprocessing will be considered complete when:

1. All three splits have been processed according to their defined roles.
2. Original images, captions, and annotations remain preserved.
3. Image and caption records maintain their correct associations.
4. Metadata has been organised and validated.
5. The CLIP and Sentence-BERT configurations have been documented.
6. The quality-control checks have passed or documented exceptions have been resolved.
7. The anti-leakage rules have been verified.
8. The resulting artifacts are organised and reproducible.

The validated preprocessing outputs will then be used for embedding generation and knowledge-base construction.