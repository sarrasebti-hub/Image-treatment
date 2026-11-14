# Open issues (before freezing the preprocessing release)

## Open

1. **Stratified test set.** Size is settled at **100** images. The brief asks for a *stratified* val2017 subset identical for all groups; ours is random (seed 42). Max category prevalence gap vs val2017 is 3.0 points, and 8 categories are absent (baseball glove, airplane, baseball bat, bear, toothbrush, scissors, hair drier, toaster) — see `results/qa/test_stratification_diagnostic.csv`. If the teachers require stratification, regenerate `split_ids.json`, `image_manifest.csv`, the test archive and the test rows of `image_metadata.*`. KB, embeddings, indexes and the validation result are unaffected.
2. **Refresh the audit record.** Run notebook 06 part B so `preprocessing_audit.json` lists all 17 checks (the committed file is from the 13-check run). Copy the corrected `data/embeddings/clip_vit_base_patch32/embedding_config.json` from this repo to Drive first (see below).

## Known limits (documented, not blocking)

- **Heuristic attributes.** Colours/shapes/context = caption keywords; relations = bbox overlap. Not visual ground truth.
- **Two metadata sources.** KB object counts are human labels; validation/test counts come from DETR (score ≥ 0.7). Not directly comparable.
- **Validation metric is a proxy** (object-category Jaccard@5, 98 scorable images). It cannot detect differences below ~0.01.
- **Near-duplicates.** Only byte-identical images are checked across splits.
- **CLIP centre crop** loses image sides; CLIP/BLIP may have seen COCO in pre-training.

## Closed (10 Nov 2026)

- ~~Validation experiment (§4.4.4)~~ → done, keep current config (`results/preprocessing_release/validation_selection.*`).
- ~~Embedding config missing image batch size; vector-alignment check~~ → done, PASS (`results/qa/embedding_alignment_check.json`).
- ~~Per-image metadata only for KB objects~~ → 10,200 records with sources (`data/metadata/image_metadata.*`).
- ~~Shape analysis~~ → shape keywords per image in metadata (6.9 % of KB images have one).
- ~~Test size 100 vs 1,000~~ → settled at 100.
- ~~Apostrophes split in BM25~~ → tested on validation, no measurable effect.
- **Note on `embedding_config.json`:** the alignment run overwrote `transformers_version` with its own version (5.22.1). The vectors were generated with **5.20.2**. The repo copy restores 5.20.2 and stores the check versions under `alignment_check`; notebook 04 no longer overwrites existing keys.

## Still not done

- Scatter plots requested in §4.3.
