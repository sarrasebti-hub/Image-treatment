# Section 4.3: Embeddings Generation, Vector Indexing & Knowledge Base Retrieval

**Role:** Member 3 – Embeddings & Knowledge Base Lead  
**Lead:** Rima AHFIR  
**Group:** Group 6 (Sparse / BM25 RAG Track)  
**Date:** November 2026  

---

## 1. Overview and Objectives
As Member 3 (Embeddings & Knowledge Base Lead), the primary objective for Phase 1 was to construct, index, and validate the dense vector search foundation over the frozen 10,000 MS COCO 2017 Knowledge Base (KB) samples.

Key responsibilities included:
- **Encoder Selection & Justification:** Selecting and integrating semantic text and visual representation models.
- **Dense Vector Extraction:** Encoding 10,000 multi-modal samples, auditing feature dimensions, and enforcing normalization.
- **Vector Database Construction:** Building a high-throughput, exact nearest-neighbor FAISS index and documenting index parameters.
- **Quality Control (QC):** Verifying the absence of NaNs, zero vectors, duplicate IDs, and text mismatch regressions.
- **Validation Retrieval Sanity Check:** Executing cross-split evaluation on the validation set (`splits/val_manifest.json`) to confirm semantic accuracy with zero data leakage.

---

## 2. Model Selection and Architecture Justification

Per project requirements, two complementary encoder architectures were evaluated:

| Component | Model Identifier | Embedding Dim | Similarity Metric | Justification |
| :--- | :--- | :---: | :---: | :--- |
| **Dense Semantic Text** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Cosine Similarity (Inner Product) | Fine-tuned explicitly on >1B sentence pairs for semantic search; compact, high-speed CPU/GPU inference, low latency, robust lexical-semantic representation. |
| **Multimodal Vision-Text** | OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`) | 512 | Cosine Similarity (Inner Product) | Trained on LAION-2B; bridges caption representations and image semantics in a shared latent hypersphere for future cross-modal text-to-image queries. |

All vectors are explicitly $L_2$-normalized ($\vert{}\vert{}\mathbf{v}\vert{}\vert{}_2 = 1.0$) prior to persistence and indexing. This maps Euclidean distance monotonically to Inner Product, enabling rapid Cosine Similarity computations ($S_C(u, v) = u \cdot v$).

---

## 3. Embedding Generation Pipeline & Quality Control

### 3.1 Input Mapping
Input data was ingested from the frozen releases delivered by Member 1 (`splits/kb_manifest.json`, `splits/kb_ids.json`) and Member 2 (`data/captions_processed.json`). Each of the 10,000 samples contains 5 distinct captions; the primary descriptive caption was extracted to represent the visual scene in the dense knowledge base.

### 3.2 Quality Control Audit
Before indexing, the generated matrices were validated against strict data integrity constraints:
- **Total Vectors Extracted:** 10,000 entries.
- **Null / NaN Check:** Exactly 0 `NaN` or `Inf` values (`assert not np.isnan(vectors).any()`).
- **Identifier Uniqueness:** 10,000 unique COCO Image IDs (`len(valid_ids) == len(set(valid_ids))`).
- **Feature Sparsity:** No dead channels or collapsed zero-vectors detected.

### 3.3 Artifact Outputs
The tensors were serialized to disk under `embeddings/`:
- `embeddings/kb_sbert_text_embeddings.npy` (Shape: `10000 x 384`, `float32`, size: ~15.36 MB)
- `embeddings/kb_clip_text_embeddings.npy` (Shape: `10000 x 512`, `float32`, size: ~20.48 MB)
- `embeddings/kb_image_ids.npy` (Shape: `10000`, `int64`)

---

## 4. FAISS Vector Database Construction

### 4.1 Index Topology
For 10,000 records, an exact Flat Inner Product index (`faiss.IndexFlatIP`) was chosen over approximate clustering (e.g., IVF or HNSW). An exact flat index guarantees 100% recall without quantization noise or hyperparameter approximations.

### 4.2 Index Metadata and Build Metrics
- **Index Type:** `faiss.IndexFlatIP`
- **Metric:** Cosine Similarity via Inner Product on unit-normalized vectors
- **Dimension ($d$):** 384 (SBERT) / 512 (CLIP)
- **Total Indexed Entries:** 10,000 vectors
- **Binary File Size:** ~15.4 MB (`kb_sbert_flat_ip.index`)
- **Indexing Construction Time:** **0.0603 seconds** (instantaneous addition to CPU RAM)

The configuration file was exported to `index/index_config.json`.

---

## 5. Retrieval Sanity Check & Leakage Verification

### 5.1 Protocol
To ensure that the frozen Knowledge Base supports downstream RAG retrieval, 5 unseen query captions from the validation split (`splits/val_manifest.json`) were queried against the vector index for Top-$k$ ($k=3$) nearest neighbors.

### 5.2 Sanity Check Results

```text
===========================================================================
Query [Val ID 9379]: "A bus stop next to a curvy road surrounded by traffic lights."
   Rank 1 (Score: 0.8295) -> KB ID 567013: "a bus on a city street next to a traffic light"
   Rank 2 (Score: 0.7428) -> KB ID 344578: "A bus stopped at a bus stop near a street."
   Rank 3 (Score: 0.7321) -> KB ID 262442: "A bus rides down the road near traffic."
---------------------------------------------------------------------------
Query [Val ID 23840]: "A plate topped with slices of banana next to a knife."
   Rank 1 (Score: 0.7723) -> KB ID 90192:  "A white plate topped with sliced up fruit."
   Rank 2 (Score: 0.7448) -> KB ID 331339: "A plate topped with waffles, bacon and banana slices."
   Rank 3 (Score: 0.7129) -> KB ID 297300: "A white plate topped with a slice of apple and a fork."
---------------------------------------------------------------------------
Query [Val ID 26383]: "A bathroom with blue walls, a toilet, and sink."
   Rank 1 (Score: 0.8408) -> KB ID 10290:  "A bathroom with a blue shower curtain and blue walls."
   Rank 2 (Score: 0.8076) -> KB ID 329286: "A bathroom with a white sink and a blue tiled shower."
   Rank 3 (Score: 0.7966) -> KB ID 54889:  "A bathroom that has a sink, and a toilet."
---------------------------------------------------------------------------
Query [Val ID 34235]: "a man is sitting with a laptop box on him"
   Rank 1 (Score: 0.7691) -> KB ID 460916: "A man sitting on a bed with a laptop."
   Rank 2 (Score: 0.7184) -> KB ID 5632:   "A man sitting on a couch using a laptop computer."
   Rank 3 (Score: 0.7150) -> KB ID 170917: "A man sitting in front of a laptop computer."
---------------------------------------------------------------------------
Query [Val ID 39697]: "A pizza is shown in a box uncovered and cooked."
   Rank 1 (Score: 0.8540) -> KB ID 171467: "A pizza is displayed in an open box."
   Rank 2 (Score: 0.8332) -> KB ID 117578: "A cooked pizza sits in a box, with meat and vegetable toppings."
   Rank 3 (Score: 0.7511) -> KB ID 339159: "The pizza is cooked and ready to be eaten"
===========================================================================

### 5.3 Index Verification (v1.1)
In addition to the validation sanity check above, the following checks were run on the final indexes.
After the caption exclusion, the indexes were verified with src/verify_index.py
(result in report/index_id_check.json). The 10,000 indexed image IDs are exactly the
kb IDs, with no duplicate; no test or validation image is in the index, so a test image
cannot be retrieved for itself. The 75 primary captions that appeared in
excluded_captions.json were replaced by another non-excluded caption of the same image,
and no indexed caption equals a test or validation caption after normalisation. Each of
the three indexes contains 10,000 vectors.
