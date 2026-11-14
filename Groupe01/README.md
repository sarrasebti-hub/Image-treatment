# RAG Image Captioning, Group G1: Phase 1 (data exploration and preprocessing)

Shared preprocessing for the comparative study of RAG techniques for image
captioning on MS COCO. Group G1 technique: Naive / Dense RAG.

**Authors:** Cerine, Nour (akhdari), Sarah (Gherbi), Maria (Saadi)

**Data and artifacts:** `artifacts/` (JSON/CSV/PNG) is in the public repository https://github.com/cerine25/rag-image-final. Embeddings and FAISS indexes are in the release `g1-v1.2` of that repository.

**Release `g1-v1.2`** — official test set: **1,000 images** (COCO val2017).
The 100-image test of the earlier frozen release `g1-v1.1` is kept unchanged as
a *legacy* set (see [Test sets](#test-sets)).

## Splits

| Split | Source | Size | Use |
|---|---|---|---|
| Knowledge base | COCO train2017 | 10,000 images, 50,000 captions | Retrieval index |
| Validation | COCO train2017 (disjoint from KB) | 100 images | Encoder / config selection |
| **Test (official)** | COCO val2017 | **1,000 images**, 5,000 reference captions | Final evaluation |
| Test (legacy) | COCO val2017 | 100 images, 500 reference captions | Kept for reference, frozen in `g1-v1.1` |

## Test sets

The project specification states 100 test images in sections 4.1/4.2 and
1,000 in section 4.4. The instructor indicated (verbally) that the test set
must contain **1,000 images**, so:

- **Official test**: `experiments/test_1000_val2017/splits/test_1000.json`
  (1,000 images, disjoint from the KB, the validation split and the legacy test).
- **Legacy test**: `artifacts/splits/test.json` (100 images) is part of the
  frozen release `g1-v1.1` and is **not modified** (SHA-256 hashes are recorded
  in `artifacts/manifest.json` and
  `experiments/test_1000_val2017/manifests/verification.json`).
- The decision is recorded in `reports/split_confirmation.json`.

Selection of the encoder/configuration is done on the **validation** split only;
neither test set is used to select a configuration.

## Anti-leakage audit

    python scripts/check_leak_test1000.py      # ID + caption leakage for the 1,000 test
    python scripts/qc_test_1000.py             # full QC report for the 1,000 test

Verified: test(1000) ∩ KB = 0, test(1000) ∩ validation = 0,
test(1000) ∩ legacy test(100) = 0, and no test reference caption
belongs to a KB image. Captions are stored in separate files:
`kb_captions.jsonl` (indexed) vs `eval_refs_*.jsonl` (evaluation only).
QC reports: `artifacts/qc/qc_report.md` (frozen `g1-v1.1`, 40 checks, French labels;
English version: `reports/qc_report_g1-v1.1_en.md`) and
`experiments/test_1000_val2017/qc/qc_report_test_1000.md` (official test).

## Results on the official test (1,000 images)

Nearest-neighbour caption retrieval, CIDEr against the 5 references
(`experiments/test_1000_val2017/results/encoder_comparison_test_1000.csv`):

| Configuration | Recall@1 | MRR | CIDEr |
|---|---:|---:|---:|
| CLIP image-to-text | 0.952 | 0.970 | 0.575 |
| CLIP image-to-image | 0.943 | 0.965 | 0.498 |
| DINOv2 image-to-image | 0.965 | 0.978 | 0.531 |
| SigLIP image-to-image | 0.966 | 0.980 | 0.532 |
| SigLIP image-to-text | 0.973 | 0.985 | **0.639** |
| Random-caption baseline | | | 0.027 |

Note: SigLIP image-to-text was the validation winner (100 images, CIDEr 0.722) and is also the best of the five on the 1,000-image test (CIDEr 0.639). These figures use SigLIP text embeddings computed with `padding="max_length"` (64 tokens), as in notebook 04. An earlier version of this table (SigLIP image-to-text CIDEr 0.464, Recall@1 0.891, MRR 0.938, lowest of the five; CLIP image-to-text highest) came from SigLIP text embeddings recomputed with `scripts/recover_kb_embeddings.py`, which uses `padding=True`; we measured that this setting reproduces exactly those earlier figures. Which of the two text-embedding sets matches the SHA256 recorded in `artifacts/manifest.json` is not established. CLIP image-to-image (0.4984 vs 0.4978 earlier) and DINOv2 image-to-image (0.5309 vs 0.5333) also changed slightly; the cause of these small differences is not established. The technical report PDF (`reports/phase1_technical_report.pdf`, section 11.3) was recompiled after this correction and shows the corrected figures (Table 10). The encoder was selected on validation and was not re-selected on the test set. Details: `RELEASE_NOTES.md`.

## Repository structure

- `00`-`09_*.ipynb`: pipeline (annotations, splits, images, preprocessing,
  dedup, captions and metadata, DETR detection, embeddings and index, EDA,
  config selection, QC, freeze, retrieval demo)
- `artifacts/`: frozen `g1-v1.1` release (splits, audit, EDA with 11 figures, tables, QC, manifest)
- `experiments/test_1000_val2017/`: official 1,000-image test (split, references,
  metadata, QC, encoder comparison)
- `scripts/`: 1,000-image pipeline, encoder comparison, leakage check, QC, release packaging
- `reports/`, `docs/`: technical report, leakage audit, dataset card,
  preprocessing documentation
- `RELEASE_AND_PR_GUIDE.md`: how to publish the data release and submit the Pull Request

## Reproduction

Python 3.12–3.13 is supported (the frozen release was built with 3.13.7).

Linux / macOS / Google Colab:

    python -m venv .venv
    source .venv/bin/activate
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\activate
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

Run the notebooks in numeric order, then for the 1,000-image test:

    python scripts/run_test_1000.py

Images, embeddings and FAISS indexes are **not stored in Git** (size).
Either regenerate them with the notebooks (`01b`, `02`, `04`) and scripts above, or
download the prebuilt data from the GitHub Release `g1-v1.2`
(see `RELEASE_AND_PR_GUIDE.md`; SHA-256 checksums are attached to the release).

Prebuilt data (Windows PowerShell), from the repository root:

    Get-FileHash g1-v1.2-embeddings.zip -Algorithm SHA256
    Get-FileHash g1-v1.2-faiss-indexes.zip -Algorithm SHA256
    Get-FileHash g1-v1.2-test1000-embeddings.zip -Algorithm SHA256
    # compare each hash with SHA256SUMS.txt attached to the g1-v1.2 release
    Expand-Archive -Path g1-v1.2-embeddings.zip -DestinationPath .
    Expand-Archive -Path g1-v1.2-faiss-indexes.zip -DestinationPath .
    Expand-Archive -Path g1-v1.2-test1000-embeddings.zip -DestinationPath .

See also `PHASE1_REPRODUCTION_GUIDE.md` and `docs/DATASET_CARD.md`.

## Contributors

Cerine Benmedghar, Nour El Houda Akhdari, Sarah Gherbi, Maria Saadi
