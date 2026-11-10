# Final scientific review: train/test pHash candidates

## Decision: WARNING

All four train/test pHash candidates were compared at original resolution. None has equal SHA-256, and the existing dataset-wide exact-hash report still records zero exact duplicate groups. Three pairs are best explained as distinct images of the same or a similar scene. Pair **535889 / 465129** is a strong near-duplicate candidate and remains unresolved as to source provenance. It is retained, flagged, and keeps Phase 1 incomplete. No image or manifest was changed.

The pHash algorithm and threshold are unchanged: 64-bit DCT pHash, candidate threshold **Hamming distance ≤6**. Distances identify review candidates; they do not by themselves prove duplicate identity. All raw files in these pairs have equal dimensions within their pair. SHA-256 inequality excludes byte-for-byte identity, but does not exclude recompressed or edited derivatives.

## Pair findings

| Train / test image IDs | pHash distance | Original SHA-256 | Comparison evidence | Classification and disposition |
|---|---:|---|---|---|
| 291597 / 525083 | 6 | Different | 640×428 each; CLIP cosine 0.9816; raw RGB MAE 28.18/255, correlation 0.7944. Same kitchen geometry and viewpoint, with substantial pixel differences. | **Same scene, distinct image** (moderate confidence). Retain; no duplicate declaration. Source-provenance uncertainty remains. |
| 58174 / 133819 | 6 | Different | 640×425 each; CLIP cosine 0.9312; raw RGB MAE 59.92/255, correlation 0.4615. Bus body/front, route display, position, and street details differ. | **Different images of a similar urban bus scene** (high confidence). Retain; no duplicate declaration. |
| 214648 / 129062 | 6 | Different | 640×480 each; CLIP cosine 0.9623; raw RGB MAE 40.50/255, correlation 0.6044. Storefront and viewpoint match, while people and foreground details differ materially. | **Distinct captures of the same place/scene** (high confidence). Retain; no duplicate declaration. |
| 535889 / 465129 | 2 | Different | 640×480 each; CLIP cosine 0.9889; raw RGB MAE 4.00/255, correlation 0.99857; 0% of pixels have a maximum-channel delta above 25. Composition and most pixels are nearly identical; small pose/detail differences are visible. | **Strong near-duplicate candidate; source relation unresolved.** Retain and flag. Could be a recompressed/derived image or a near-identical capture; current metadata cannot determine which. |

Raw pixel metrics are descriptive comparisons of same-size, unregistered files. They are sensitive to encoding, color, and exposure. CLIP cosine is a semantic similarity signal, not a duplicate classifier. COCO category overlap and caption similarity were reviewed as context only; neither was used to declare identity.

## Evidence artifacts

- `reports/perceptual_hash/cross_split_native_resolution_comparison.png` — all four original-resolution pairs.
- `reports/perceptual_hash/hamming2_native_2x.png` — magnified comparison of the Hamming-2 pair.
- `reports/perceptual_hash/hamming2_absolute_difference_x8.png` — 8×-amplified absolute pixel difference for that pair.
- `reports/perceptual_hash/cross_split_pair_evidence.json` — dimensions, SHA-256 values, CLIP similarities, raw pixel metrics, reference captions, and category context.
- `reports/perceptual_hash/cross_split_visual_review.json` — pair-level classification, rationale, confidence, and unresolved uncertainty.

The pre-existing `duplicate_candidates.json`, `hash_statistics.json`, `phash_database.npz`, and `selected_images_integrity.json` were preserved. The four candidate pairs remain in the frozen train/test splits as required. No silent replacement or removal was made.

## Quality-control result

**WARNING** — the audit itself completed without errors, but pair 535889 / 465129 remains a cross-split near-duplicate risk. Phase 1 must remain **INCOMPLETE** until the project owner decides whether this pair is acceptable or authorizes a versioned split change. The other three candidates are documented as distinct images depicting the same or similar scenes, not duplicates.
