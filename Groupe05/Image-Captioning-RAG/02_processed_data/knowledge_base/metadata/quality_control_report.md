# Knowledge Base Quality Control Report

## 1. General Information

| Property | Value |
|---|---|
| Project | MS-COCO Image Captioning Knowledge Base |
| Dataset | MS-COCO 2017 |
| Knowledge-base split | Training |
| Random seed | 45 |
| Report version | 1.0 |
| Generated at (UTC) | 2026-11-06T00:51:15.183061+00:00 |
| Overall status | Passed |

## 2. Dataset Summary

| Dataset component | Count |
|---|---:|
| Training images | 9,900 |
| Validation images | 100 |
| Test images | 100 |
| Training captions | 49,500 |
| Captions per training image | 5 |

## 3. Embedding Integrity

### 3.1 CLIP Image Embeddings

| Property | Result |
|---|---|
| Model | openai/clip-vit-base-patch32 |
| Number of vectors | 9,900 |
| Dimension | 512 |
| Data type | float32 |
| NaN values | 0 |
| Infinite values | 0 |
| Zero vectors | 0 |
| Normalization | L2 normalized |
| Status | PASSED |

### 3.2 SBERT Caption Embeddings

| Property | Result |
|---|---|
| Model | sentence-transformers/all-MiniLM-L6-v2 |
| Number of vectors | 49,500 |
| Dimension | 384 |
| Data type | float32 |
| NaN values | 0 |
| Infinite values | 0 |
| Zero vectors | 0 |
| Normalization | L2 normalized |
| Status | PASSED |

## 4. Mapping and Document Consistency

| Check | Result |
|---|---|
| Mapping uniqueness and row alignment | PASSED |
| Embedding-to-document consistency | PASSED |
| Image-caption relationships | PASSED |
| Image documents | 9,900 |
| Caption documents | 49,500 |
| Total caption references | 49,500 |
| Unique caption references | 49,500 |
| Unreferenced captions | 0 |
| Duplicate caption references | 0 |

## 5. Dataset Leakage Audit

| Check | Number of overlaps |
|---|---:|
| Training-validation | 0 |
| Training-test | 0 |
| Validation-test | 0 |
| Validation images in knowledge base | 0 |
| Test images in knowledge base | 0 |
| Non-training images in knowledge base | 0 |

**Result: PASSED.**

The knowledge base contains only training images and their associated captions.
Validation and test data are excluded.

## 6. FAISS Index Integrity

| Property | CLIP Image Index | SBERT Caption Index |
|---|---:|---:|
| Index type | IndexFlatIP | IndexFlatIP |
| Vector count | 9,900 | 49,500 |
| Dimension | 512 | 384 |
| Metric | Inner Product | Inner Product |
| Original vectors match | Yes | Yes |
| Status | PASSED | PASSED |

## 7. Metadata and Manifest

The knowledge-base metadata and manifest were checked against the actual artifacts.

- Dataset and split information: PASSED
- Index metadata: PASSED
- Document metadata: PASSED
- Data exclusion metadata: PASSED
- Manifest statistics: PASSED
- Artifact inventory and file sizes: PASSED

## 8. Statistical and Visualisation QC

| Artifact category | Verified files |
|---|---:|
| Figures | 5 |
| Statistical JSON files | 5 |
| CSV tables | 4 |

Visual inspection of the principal figures was completed.

Result: PASSED.

## 9. Overall Quality Control Results

| QC category | Status |
|---|---|
| Required artifact verification | PASSED |
| Embedding integrity | PASSED |
| Mapping uniqueness and alignment | PASSED |
| Embedding-to-document consistency | PASSED |
| Image-caption relationships | PASSED |
| Dataset leakage audit | PASSED |
| FAISS index integrity | PASSED |
| Metadata and manifest consistency | PASSED |
| Statistical artifact verification | PASSED |
| Visual inspection | PASSED |

## 10. Retrieval Pipeline Compatibility Validation

### Frozen-Release Compatibility Smoke Test

**Status: PASSED**

The frozen knowledge-base release was successfully tested with a minimal CLIP-based retrieval pipeline.

#### Query Configuration

| Property | Value |
|---|---|
| Query image ID | 250 |
| Query split | Validation |
| Excluded from knowledge base | Yes |
| CLIP model | openai/clip-vit-base-patch32 |
| Embedding dimension | 512 |
| Embedding data type | float32 |
| L2 normalization | Applied |

#### Frozen Index Configuration

| Property | Value |
|---|---|
| Index type | IndexFlatIP |
| Similarity metric | Inner product |
| Number of indexed vectors | 9,900 |
| Embedding dimension | 512 |

#### Retrieval Results

| Rank | Retrieved image ID | Similarity score |
|---:|---:|---:|
| 1 | 11,605 | 0.8479 |
| 2 | 401,670 | 0.8389 |
| 3 | 486,769 | 0.8303 |

#### Validation Checks

| Check | Status |
|---|---|
| Query image encoding | Passed |
| FAISS index search | Passed |
| Image-document mapping | Passed |
| Caption-document mapping | Passed |
| Validation/training separation | Passed |
| Associated captions verified | 15 |

#### Conclusion

The compatibility test confirms that the frozen CLIP encoder, FAISS index, image documents, and caption documents work together successfully.

A validation image can be encoded using the original CLIP model, searched against the frozen training index, and mapped to the corresponding training images and their associated human captions.

**Scope limitation:** This is a compatibility smoke test only. It does not constitute a retrieval-quality evaluation or a comparison of different RAG techniques.

---

## Overall Quality-Control Conclusion

The embeddings, mappings, documents, FAISS indexes, metadata, manifest, dataset relationships, leakage constraints, and frozen-release retrieval compatibility have been verified.

**Overall status: PASSED**

The frozen knowledge base is ready for the next development stage.
