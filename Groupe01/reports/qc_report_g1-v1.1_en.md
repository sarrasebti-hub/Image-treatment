# QC — g1-v1.1 (English view)

**PASS** — 40 checks, 0 failure(s), 0 warning(s).

_Translated view of the frozen `artifacts/qc/qc_report.json` (French labels), which is hashed in the release manifest and left unchanged._

| # | Status | Check | Detail |
|---:|---|---|---|
| 1 | PASS | KB split size | 10000 vs 10000 |
| 2 | PASS | KB ID hash = splits_meta |  |
| 3 | PASS | validation split size | 100 vs 100 |
| 4 | PASS | validation ID hash = splits_meta |  |
| 5 | PASS | test split size (legacy, 100 images) | 100 vs 100 |
| 6 | PASS | test ID hash = splits_meta |  |
| 7 | PASS | splits pairwise disjoint |  |
| 8 | PASS | invalid images (corrupt/missing) | 0 |
| 9 | PASS | unique IDs in images.csv |  |
| 10 | PASS | exact duplicates (sha256) across/within splits | 0 images |
| 11 | PASS | no NaN in pixel statistics |  |
| 12 | PASS | KB: 5 captions per image | min=5 max=5 |
| 13 | PASS | validation: 5 references per image |  |
| 14 | PASS | test: 5 references per image |  |
| 15 | PASS | unique caption_id (KB) |  |
| 16 | PASS | ANTI-LEAK: no val/test image in kb_captions |  |
| 17 | PASS | ANTI-LEAK: no evaluation caption_id in the KB |  |
| 18 | PASS | raw captions unchanged (raw != clean allowed, raw non-empty) |  |
| 19 | PASS | info: validation references textually identical to a KB caption | 1.0% — normal for generic captions; not an image leak |
| 20 | PASS | info: test references textually identical to a KB caption | 1.4% — normal for generic captions; not an image leak |
| 21 | PASS | metadata for all images |  |
| 22 | PASS | text attributes for the KB only |  |
| 23 | PASS | embeddings clip-b32/kb | shape=(10000, 512) |
| 24 | PASS | embeddings clip-b32/val | shape=(100, 512) |
| 25 | PASS | embeddings clip-b32/test | shape=(100, 512) |
| 26 | PASS | index clip-b32: ntotal = KB size and self-retrievability |  |
| 27 | PASS | embeddings siglip-b16/kb | shape=(10000, 768) |
| 28 | PASS | embeddings siglip-b16/val | shape=(100, 768) |
| 29 | PASS | embeddings siglip-b16/test | shape=(100, 768) |
| 30 | PASS | index siglip-b16: ntotal = KB size and self-retrievability |  |
| 31 | PASS | embeddings dinov2-s/kb | shape=(10000, 384) |
| 32 | PASS | embeddings dinov2-s/val | shape=(100, 384) |
| 33 | PASS | embeddings dinov2-s/test | shape=(100, 384) |
| 34 | PASS | index dinov2-s: ntotal = KB size and self-retrievability |  |
| 35 | PASS | ANTI-LEAK: validation image never retrieved as itself (k=10) |  |
| 36 | PASS | info: validation images with a near-duplicate in the KB | cos>=0.97: 0/100 / dHash<=4: 0 |
| 37 | PASS | ANTI-LEAK: test image never retrieved as itself (k=10) |  |
| 38 | PASS | info: test images with a near-duplicate in the KB | cos>=0.97: 0/100 / dHash<=4: 0 |
| 39 | PASS | smoke test val: nearest-neighbour caption CIDEr > random | NN=0.674 random=0.024 / Jaccard classes@1=0.546 |
| 40 | PASS | smoke test test: nearest-neighbour caption CIDEr > random | NN=0.622 random=0.033 / Jaccard classes@1=0.506 |
