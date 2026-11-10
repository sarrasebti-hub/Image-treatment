# Final scientific audit ? Phase 1 controlled revision

## Decision: PASS

Phase 1 passes the final quality-control run for controlled dataset revision 1.1.0. QC reports **PASS**, `phase1_complete: true`, with zero warning, failed, or not-run checks. The four previously reviewed test IDs were deliberately replaced because one train/test pair had strong near-duplicate evidence and the three others were also flagged candidates requiring split separation under this task. The replacement was limited to test IDs; train and validation IDs were not changed. All seven original manifest and selection-report files are preserved byte-for-byte in `data/selected/backups/phase1_pre_test_replacement_v1/`, and the revision record records before/after SHA-256 for each.

No source or processed image was deleted or moved. The old raw and processed image files remain present. Existing pre-revision SHA-256 and pHash results are preserved byte-for-byte in the backup. Current pHash and exact-SHA audits were recomputed for the revised selection.

## Four train/test pairs reviewed

All four pair comparisons used original-resolution image inspection, SHA-256, 64-bit DCT pHash, same-coordinate pixel metrics, and cached CLIP cosine. pHash, CLIP, and pixel similarity are screening/evidence signals, not identity proof. Unequal SHA-256 proves the files are not byte-identical, but cannot exclude recompression or editing.

| Train / prior test ID | pHash Hamming | Evidence | Classification | Disposition |
|---|---:|---|---|---|
| 291597 / 525083 | 6 | SHA-256 unequal; 640?428 each; CLIP cosine 0.9816; raw RGB MAE 28.18/255, correlation 0.7944. Kitchen geometry and viewpoint are very similar, with substantial pixel differences. | Same scene, distinct image (moderate confidence); not called a duplicate. | Test ID 525083 replaced with 195754. Retained only as historical evidence; no current cross-split pair. |
| 58174 / 133819 | 6 | SHA-256 unequal; 640?425 each; CLIP cosine 0.9312; MAE 59.92/255, correlation 0.4615. Bus design, position, route display, and street details differ. | Different images depicting a similar bus/street scene (high confidence); not duplicates. | Test ID 133819 replaced with 165039. |
| 214648 / 129062 | 6 | SHA-256 unequal; 640?480 each; CLIP cosine 0.9623; MAE 40.50/255, correlation 0.6044. The storefront and viewpoint match; people and foreground details differ. | Distinct captures of the same place/scene (high confidence); not duplicates. | Test ID 129062 replaced with 180487. |
| 535889 / 465129 | 2 | SHA-256 unequal; 640?480 each; CLIP cosine 0.9889; MAE 4.00/255, correlation 0.99857; 0% of pixels exceed max-channel delta 25. Same composition and near-identical pixel structure, with small visible pose/detail differences. | Strong near-duplicate; source relation unresolved. Could be a recompressed/edited derivative or an adjacent near-identical capture. | Test ID 465129 replaced with 294350. The unresolved provenance issue no longer occurs across current train/test selection. |

Pixel metrics are descriptive and sensitive to compression, alignment, color, and exposure. CLIP scores and caption/category overlap are contextual signals, not duplicate classifiers. Similar-scene cases were explicitly kept distinct in classification.

## Replacement selection and retained uncertainty

Four unused val2017 images with exactly five linked captions were selected from the screened eligible pool (3,887 images). The deterministic replacement seed was 20261107. Candidates were decoded and their processed-equivalent pHashes compared against 10,100 train/validation hashes; minimum distances were respectively 16, 10, 12, and 16 for replacements 195754, 165039, 180487, and 294350, all greater than the inclusive candidate threshold of 6. No candidate had an exact SHA-256 match to a surviving selected source image. Replacement selection matched annotation-derived complexity strata and optimized category overlap and other documented annotation proxies; these are sampling proxies, not semantic equivalence claims.

The pre-revision d=2 pair's provenance cannot be resolved from COCO data or image bytes alone. It is retained as historical uncertainty, with both images removed from the current cross-split pairing by the controlled test revision. The other three pairs were similar scenes, not duplicate images, and were not labeled duplicates.

## Current revision verification

- Current selection: 10,000 train, 100 validation, 1,000 test; no split overlap; 5 captions per selected image.
- Current exact SHA-256 audit: zero duplicate groups.
- Current pHash: all 11,100 selected images hashed; 9 within-split candidates and **0 cross-split candidates** at Hamming distance ?6. The nine within-split candidates are not cross-split leakage; they remain pHash candidates rather than asserted duplicates.
- Current image embeddings map to manifest order; retrieval, caption index, POS, and visual outputs passed QC.
- All seven pre-revision manifest/report files match their archived SHA-256; train/validation current files also match their original hashes. Test IDs and selection report changed intentionally and match the revision record.
- All original four raw images and their processed test images remain on disk. No image was deleted or moved.
- Original pHash database/statistics/candidate and exact-SHA report are archived byte-for-byte; fresh current-revision outputs are stored at their standard report paths.

Detailed per-pair metrics are preserved in `reports/perceptual_hash/cross_split_pair_evidence.json`; the prior classification and current empty cross-split review are in `reports/perceptual_hash/cross_split_visual_review.json`. Replacement IDs, SHA-256, selection rationale, and before/after manifest hashes are in `data/selected/replacement_history/revision_1.json` and `reports/selection_report/replacement_candidate_screening.json`.
