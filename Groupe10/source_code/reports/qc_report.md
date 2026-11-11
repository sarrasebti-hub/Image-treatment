# Quality Control Report - Embeddings & Vector Database

**Generated:** 2026-11-10T20:11:38.950059+00:00  
**Task:** Yahia - Embeddings, Vector Database & Quality Control  
**Overall verdict:** **PASS WITH WARNINGS**

## 1. Configuration

| Parameter | Value |
|---|---|
| Model | openai/clip-vit-base-patch32 |
| Embedding dimension | 512 |
| Similarity metric | cosine (inner product on L2-normalized vectors) |
| Knowledge Base split | knowledge_base |
| Knowledge Base images | 10000 |
| Knowledge Base captions | 50028 |
| Evaluation split | test |
| Evaluation images | 1000 |
| Evaluation captions | 5002 |
| Index type | flat |
| Seed | 42 |

## 2. Embedding integrity

| Vectors | Count | Dim | Finite | Zero | Norm mean | Max norm dev. | Exact dups | Near dups |
|---|---|---|---|---|---|---|---|---|
| images | 10000 | 512 | True | 0 | 1.0 | 0.0 | 0 | 0 |
| captions | 50028 | 512 | True | 0 | 1.0 | 0.0 | 266 | 3 |

Near-duplicate check: cosine threshold 0.999, sample size 2000.

### Caption text statistics

| Metric | Value |
|---|---|
| Empty captions | 0 |
| Words min | 7 |
| Words mean | 10.4567 |
| Words median | 10.0 |
| Words max | 46 |

## 3. Metadata and coverage

| Check | Value |
|---|---|
| Images | 10000 |
| Captions | 50028 |
| Unique image IDs | 10000 |
| Captions referencing unknown images | 0 |
| Images without captions | 0 |
| Average captions per image | 5.0028 |
| Duplicate caption texts | 566 |
| Duplicate caption rate | 0.0113 |
| Missing image files | 0 |
| Git LFS pointer files | 0 |

Captions per image: 5 caption(s): 9972 image(s), 6 caption(s): 28 image(s)

## 4. Vector database integrity

| Index | Type | Vectors | Metadata rows | Dim | Self top-1 | Size |
|---|---|---|---|---|---|---|
| images | IndexFlatIP | 10000 | 10000 | 512 | 1.0 | 20000.0 KB |
| captions | IndexFlatIP | 50028 | 50028 | 512 | 0.99 | 100056.0 KB |
| joint | IndexFlatIP | 60028 | 60028 | 512 | 0.98 | 120056.0 KB |

Self top-1 is the fraction of sampled vectors whose nearest neighbour is the vector itself (sanity check of the index).

## 5. Retrieval quality

### Held-out evaluation split

| Direction | Queries | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|---|
| Text -> image | 200 | 0.49 | 0.795 | 0.885 | 0.6167 | 2.0 |
| Image -> text | 200 | 0.74 | 0.9 | 0.96 | 0.8022 | 1.0 |

### Knowledge Base self-retrieval

| Direction | Queries | R@1 | R@5 | R@10 | MRR | Median rank |
|---|---|---|---|---|---|---|
| Text -> image | 200 | 0.215 | 0.38 | 0.465 | 0.2803 | 11.0 |
| Image -> text | 200 | 0.355 | 0.6 | 0.735 | 0.4628 | 3.0 |

## 6. Performance

| Operation | Mean (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| CLIP text encoding | 40.32 | 39.31 | 55.98 | 21.57 | 61.02 |
| FAISS search (top-5) | 14.38 | 13.29 | 21.24 | 9.05 | 24.29 |

Measured on 20 queries (cpu).

## 7. Verdict

**Overall: PASS WITH WARNINGS**

**Warnings**

- captions: 266 exact duplicate vector(s)
- captions: 3 near-duplicate pair(s) (cosine >= 0.999)

> **Note:** duplicate caption vectors come from duplicate caption texts in COCO (the same sentence is sometimes reused for an image). They are reported for transparency and do not block the pipeline.

## 8. Limitations and recommendations

- CLIP ViT-B/32 is used **zero-shot** (no fine-tuning on COCO). Retrieval scores are therefore a baseline, not a ceiling.
- Recall@K on a small gallery is optimistic: with a 100-image gallery, random chance is already 1%. Compare runs with the same gallery size.
- The repository originally stores processed images as Git LFS pointers. The pipeline downloads the real COCO images from `coco_url` when `--download` is used; this requires an internet connection.
- For production, consider fine-tuning CLIP on COCO captions, using a larger encoder (ViT-L/14) and an approximate index (IVF/HNSW) for hundreds of thousands of vectors.
- Re-run the full Knowledge Base (`--kb-limit 0`) before reporting final numbers to the team.
