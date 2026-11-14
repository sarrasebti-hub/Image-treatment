# Phase 1 Report — Dataset Preparation and Knowledge Base Construction

## Project Title

**A Comparative Study of RAG Techniques for Generating Image Captions**



---

## 1. Introduction

The objective of Phase 1 was to prepare a structured and validated knowledge base for the image-captioning RAG project.

This phase included dataset auditing, dataset splitting, statistical exploration, image and caption preprocessing, metadata extraction, embedding generation, vector indexing, and quality control.

The resulting knowledge base contains 10,000 COCO images, five captions per image, image embeddings, text embeddings, metadata, and FAISS indexes.

## 2. Dataset and Data Splitting

The project uses the Microsoft COCO 2017 dataset.

The dataset was audited before processing to verify image availability, annotation consistency, image identifiers, and caption references.

### Dataset audit results

| Dataset   |  Images | Captions | Object annotations |
| --------- | ------: | -------: | -----------------: |
| train2017 | 118,287 |  591,753 |            860,001 |
| val2017   |   5,000 |   25,014 |             36,781 |

The audit found no missing image files, unreadable images, or unknown image references in the inspected annotations.

### Dataset splits

Following the professor's clarification, the project uses:

| Split          | Source    | Number of images | Purpose                                 |
| -------------- | --------- | ---------------: | --------------------------------------- |
| Knowledge Base | train2017 |           10,000 | Retrieval knowledge base                |
| Validation     | train2017 |              100 | Validation and configuration assessment |
| Testing        | val2017   |            1,000 | Final evaluation                        |

The random seed used for split creation was **42**.

The knowledge-base and validation images were selected separately from train2017. The test images come from val2017.

The test set was divided into four scene-complexity groups:

- Simple: 250 images
- Medium: 250 images
- Complex: 250 images
- Very complex: 250 images

The split files were saved separately, and overlap checks were performed.

## 3. Knowledge Base Construction

A structured JSONL knowledge base was created using the 10,000 selected training images.

Each record contains:

- Image ID
- Image filename
- Image dimensions
- Five associated captions
- COCO object annotations

Exactly five captions were retained per image. Where an image had more than five available captions, five were selected using a reproducible random procedure with seed 42.

### Final knowledge-base statistics

| Property           | Result |
| ------------------ | -----: |
| Images             | 10,000 |
| Captions           | 50,000 |
| Object annotations | 72,062 |
| Captions per image |      5 |

The original captions were preserved, and a separate cleaned-caption dataset was generated for retrieval and text processing.

## 4. Statistical Exploration

Statistical exploration was performed on the knowledge base to understand its content and complexity.

The following analyses were completed:

### 4.1 Object Category Distribution

The frequency of COCO object categories was calculated.

The most frequent categories included:

- Person: 22,085 annotations
- Car: 3,619 annotations
- Chair: 3,180 annotations
- Bottle: 2,045 annotations
- Book: 2,003 annotations

A top-50 object-category table and graph were generated.

### 4.2 Scene Complexity

Scene complexity was estimated using the number of object annotations per image.

| Category     | Number of images |
| ------------ | ---------------: |
| Simple       |            3,064 |
| Medium       |            2,025 |
| Complex      |            2,567 |
| Very complex |            2,344 |

The average number of annotated objects per image was approximately 7.21.

The complexity thresholds were operational categories defined for this project.

### 4.3 Caption Statistics

The caption dataset contains 50,000 captions.

Main results:

- Average caption length: approximately 10.47 words
- Vocabulary size: 9,894 unique words
- Total token count used in POS analysis: 524,354

Caption length and vocabulary statistics were generated.

### 4.4 Part-of-Speech Analysis

The spaCy English model was used to analyze grammatical categories in the captions.

The most frequent categories included:

- Nouns: 181,098
- Determiners: 107,060
- Adpositions: 88,475
- Verbs: 51,207
- Adjectives: 37,235

### 4.5 Spatial Relationship Analysis

Spatial relationships were estimated using COCO bounding boxes.

The analysis considered relationships such as:

- Left of
- Right of
- Above
- Below
- Overlapping

These relationships are geometric estimates derived from bounding-box positions and overlap, not manually verified semantic relationships.

### 4.6 Attributes and Context Analysis

Caption text was analyzed to identify mentions of:

- Colours
- Shapes
- Indoor and outdoor contexts
- Daytime and nighttime

These results represent lexical mentions in captions. They should not be interpreted as independent visual detection results.

### 4.7 Object Co-occurrence

Object co-occurrence was calculated by measuring how frequently object categories appeared together in the same image.

A co-occurrence matrix and corresponding graph were generated.

The results help describe relationships between object categories in the knowledge base.

## 5. Image and Caption Preprocessing

### 5.1 Image Inspection

All 10,000 knowledge-base images were inspected.

Results:

- 10,000 readable JPEG images
- 9,982 RGB images
- 18 grayscale images
- 0 unreadable images
- 0 detected EXIF orientation metadata cases

### 5.2 Image Preprocessing

The images were converted to RGB and saved as separate preprocessed copies.

The original images were not modified.

Image resizing, cropping, rescaling, and normalization were handled by the selected CLIP processor during embedding generation.

### 5.3 Caption Preprocessing

A cleaned version of the captions was generated.

The preprocessing operations included:

- Lowercasing
- Punctuation removal
- Whitespace normalization

The original captions were preserved separately.

Results:

- 10,000 image records
- 50,000 original captions
- 50,000 cleaned captions

### 5.4 Metadata Extraction

Metadata was prepared for all 10,000 knowledge-base images.

The metadata includes:

- Image ID and filename
- Object annotations and categories
- Bounding-box information
- Estimated spatial relationships
- Caption-based colour and shape mentions
- Caption-based context mentions

COCO annotations were used as the object source. Attribute and context information was derived from caption mentions.

The metadata is linked to the corresponding image identifiers.

## 6. Embedding Generation

The selected model was:

**CLIP ViT-B/32 — `openai/clip-vit-base-patch32`**

The same CLIP checkpoint was used for image and text encoding.

### Embedding configuration

| Property          | Image embeddings | Text embeddings |
| ----------------- | ---------------- | --------------- |
| Model             | CLIP ViT-B/32    | CLIP ViT-B/32   |
| Number of vectors | 10,000           | 50,000          |
| Dimension         | 512              | 512             |
| Data type         | float32          | float32         |
| Batch size        | 32               | 64              |
| Normalization     | L2               | L2              |

Embedding generation was performed in Google Colab using an NVIDIA Tesla T4 GPU.

The resulting vectors were checked for:

- Correct dimensions
- Finite values
- L2 normalization
- Correct image and caption mappings
- Correct correspondence with the knowledge-base records

All 11 combined embedding validation checks passed.

## 7. Vector Database and Index Construction

FAISS was selected as the vector indexing library.

Two separate indexes were constructed.

### 7.1 Image Index

- Index type: `IndexFlatIP`
- Number of vectors: 10,000
- Vector dimension: 512
- Similarity: Inner Product
- Vectors: L2-normalized

### 7.2 Text Index

- Index type: `IndexFlatIP`
- Number of vectors: 50,000
- Vector dimension: 512
- Similarity: Inner Product
- Vectors: L2-normalized

Because the vectors are L2-normalized, inner-product ranking is equivalent to cosine-similarity ranking.

`IndexFlatIP` performs exact nearest-neighbour search rather than approximate search.

### 7.3 Indexing Performance

| Index | Number of vectors |  Indexing time |
| ----- | ----------------: | -------------: |
| Image |            10,000 | 0.0619 seconds |
| Text  |            50,000 | 0.1685 seconds |
| Total |            60,000 | 0.2304 seconds |

The indexes were saved to disk and successfully reloaded.

Search tests using stored vectors returned the expected top result with similarity values close to 1.0.

## 8. Metadata and Mapping Integration

The metadata records were linked to the image and caption indexes using image IDs and caption indices.

Validation confirmed:

- 10,000 image vectors linked to 10,000 metadata records
- 50,000 caption vectors linked to their caption metadata
- All caption mappings point to existing image metadata
- All caption indices are valid
- Five caption positions exist for each knowledge-base image

This establishes the metadata and mapping foundation required for the assigned Metadata-Filtered RAG approach.

The actual metadata-filtered retrieval strategy will be implemented in the subsequent project phase.

## 9. Quality Control

A final quality-control procedure was performed on the generated embeddings, indexes, mappings, and metadata.

The final Colab quality-control report checked 19 conditions.

All 19 checks passed.

The checks included:

- Embedding dimensions
- Finite embedding values
- Vector normalization
- FAISS index dimensions and vector counts
- Unique image identifiers
- Image and caption mapping consistency
- Metadata availability
- Required file availability
- Image and text index self-retrieval
- Saved-index reload verification

A separate local quality-control report also passed with zero errors and zero warnings.

### Final QC result

**Status: PASSED — 19/19 checks**

The final report was saved as `final_phase1_qc_report.json`.

## 10. Deliverables

The Phase 1 deliverables include:

- Dataset audit report
- Dataset split files and metadata
- Structured knowledge base
- Cleaned captions
- Statistical tables and graphs
- Preprocessed images
- Image metadata
- Image embeddings
- Text embeddings
- Image and text mapping files
- FAISS image index
- FAISS text index
- Indexing configuration and performance information
- Final quality-control report
- Phase 1 release archive

The embedding and indexing release is packaged in `phase1_release.zip`.

## 11. Limitations and Notes

The following points must be considered when interpreting the results:

1. Spatial relationships are estimated from bounding boxes and are not manually verified semantic relationships.
2. Colour, shape, and context statistics are based on caption mentions rather than independent visual attribute detection.
3. The selected CLIP processor handles image resizing and normalization during embedding generation.
4. The FAISS indexes use exact inner-product search over normalized vectors.
5. Metadata linkage and index self-retrieval were validated. The complete metadata-filtered RAG retrieval pipeline is part of the subsequent implementation phase.
6. The 100-image validation split was created separately from the knowledge base. Any claim that it selected the optimal preprocessing configuration should be made only if the corresponding comparison experiments were actually performed.

## 12. Conclusion

Phase 1 established a structured and validated multimodal knowledge base for the image-captioning RAG project.

The final dataset contains 10,000 images, 50,000 captions, 10,000 image embeddings, and 50,000 text embeddings. Metadata was extracted and linked to the corresponding images and captions. Separate FAISS indexes were constructed for image and text retrieval.

The final embedding and index quality-control procedure passed all 19 checks.

The resulting release provides the data, embeddings, mappings, metadata, and vector indexes required to proceed with the implementation and evaluation of the assigned Metadata-Filtered RAG approach.
