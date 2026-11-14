# G9 — Preprocessing release candidate
This release has not been frozen. See preprocessing_audit.json.
COCO: 10,000 KB + 100 validation + 100 random test images.
KB: 50,000 cleaned retrieval captions and original captions kept.
Image & text: CLIP ViT-B/32, 512 dimensions, L2-normalized.
Indexes: FAISS Flat inner-product + BM25.
Test split is NOT yet confirmed as a shared stratified evaluation subset.
Caption cue statistics are heuristics, not visual ground truth.
BBox spatial proxies are geometric, not semantic object relations.

## Automated checks
- split_counts: PASS
- split_disjoint: PASS
- manifest_ids: PASS
- captions_five_each: PASS
- kb_annotations_scoped: PASS
- image_preprocessing_pass: PASS
- image_embedding_shape: PASS
- text_embedding_shape: PASS
- embeddings_finite_and_normalized: PASS
- faiss_index_counts: PASS
- bm25_exists: PASS
- archive_manifest_match: PASS
- no_exact_cross_split_duplicates: PASS

## Pending
- Professor-confirmed shared stratified TEST IDs (PDF says both 100 and 1000)
- Confirm context/attribute annotations if professor requires visually verified labels
- Perceptual near-duplicate detection (exact byte duplicates checked)

Do not change split IDs or overwrite embeddings/indexes until all teams agree on the official split.
