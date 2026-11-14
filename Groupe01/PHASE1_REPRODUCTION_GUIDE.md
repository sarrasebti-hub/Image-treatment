# Phase 1 Reproduction Guide

Repository: `gherbisarah6-dot/rag-image-captioning-phase1`

Release documented by this repository: **g1-v1.1**

This guide describes the workflow implemented in the repository. It is not a generic COCO or RAG tutorial. It distinguishes the original **Cerine v1.0 baseline** from **Sarah v1.1 additions** and warns about the frozen release.

## Important safety rules

- Do **not** run notebook stages against the frozen `artifacts/` directory if the purpose is simply to inspect the delivered release.
- Do **not** run the reproducible runner against an unmodified frozen release: it may unlock, regenerate, or overwrite artifacts.
- Create a separate copy of the repository for a from-scratch reproduction.
- Preserve the original `artifacts/` directory as reference.
- `01_make_splits`, `01b_download_images`, `02_preprocess_images`, and later stages write files under `artifacts/`.
- The freeze notebook marks `artifacts/` files read-only. Before deliberately rebuilding, use a separate working copy or deliberately reset permissions in that copy.

## Chronological workflow

### 1. Start from a clean working copy

**RUN:**

```powershell
git clone https://github.com/gherbisarah6-dot/rag-image-captioning-phase1.git
cd rag-image-captioning-phase1
git switch sarah-phase1
```

If `sarah-phase1` is already the checked-out branch, the clone and switch can be omitted.

**INSPECT:** Confirm the repository contains:

- `config.yaml`
- `requirements.txt`
- `common.py`
- `rag_data.py`
- `00_download_annotations.ipynb` through `09_demo_retrieval.ipynb`
- `tools/run_notebooks.py`

**VERIFY:**

```powershell
git branch --show-current
git status --short
```

The reproduction should start from a clean worktree. Git may itself report a populated frozen release; that is not a failure.

**WHY:** The pipeline writes outputs and may change file permissions. Starting in a copy protects the delivered release.

---

### 2. Create the Python environment

The v1.1 frozen manifest was produced under:

| Component | Version |
|---|---|
| Python | 3.13.7 |
| NumPy | 2.5.4 |
| Pillow | 13.0.0 |
| PyTorch CPU | 2.15.0+cpu |
| Transformers | 5.22.1 |
| FAISS CPU | 1.15.1 |
| sentence-transformers | 6.2.0 |

The repository's [requirements.txt](/C:/Users/admin/rag-image-captioning-phase1/requirements.txt) contains the exact pinned versions.

`SIGLIP_ENV_NOTE.txt` documents the actual Sarah SigLIP run. The manifest records Python 3.13.7 even though the note says Python 3.13.3 for the SigLIP execution environment.

The repository also contains `requirements_v1.0_backup.txt`, the Cerine baseline dependency ranges. Do not use it for v1.1 reproduction unless you deliberately reproduce v1.0 exactly.

**RUN:**

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, activation is optional. You can call `.venv\Scripts\python.exe` directly, as documented for SigLIP, QC, and freeze.

**VERIFY:**

```powershell
python --version
python -c "import numpy,pandas,scipy,PIL,torch,transformers,faiss,sentence_transformers; print('dependencies OK')"
```

**Dependencies:** None.

**Storage note:** Only the repository explicitly documents avoiding the 18 GB COCO train archive: it downloads selected image files rather than the complete archive.

---

### 3. Configure the reproduction

**INSPECT:** `config.yaml`

Key settings are:

```yaml
seed: 42
release_version: "g1-v1.1"

sizes:
  kb: 10000
  val: 100
  test: 100

image:
  max_side: 512
  jpeg_quality: 95
  dhash_size: 8
  near_dup_hamming: 4

primary_encoder: siglip-b16
```

The repository states paths relative to the repository root:

```text
data/raw
artifacts
```

**VERIFY:** The frozen release g1-v1.1 (`artifacts/`) has a 100-image *legacy* test (see `reports/split_confirmation.json`); the **official** 1,000-image test is in `experiments/test_1000_val2017/`.

The project specification is inconsistent (100 test images in sections 4.1/4.2, 1,000 in section 4.4). The instructor indicated verbally that the test set must contain 1,000 images (no written confirmation on record). The official evaluation uses the 1,000-image test; the 100-image test is kept unchanged as a legacy set. Reproduce the official test with `python scripts/run_test_1000.py` and check it with `python scripts/qc_test_1000.py`.

**Dependencies:** Environment setup.

---

### 4. Download COCO annotations

**Purpose:** Provide captions and instance annotations for both COCO splits.

**Inputs:**

`00_download_annotations.ipynb` downloads the archive:

```text
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

Expected extracted files:

```text
data/raw/annotations/captions_train2017.json
data/raw/annotations/captions_val2017.json
data/raw/annotations/instances_train2017.json
data/raw/annotations/instances_val2017.json
```

**RUN:**

```powershell
jupyter lab
```

Set `CONFIG` to `config.yaml` in the first notebook configuration cell, then run `00_download_annotations.ipynb`.

Alternatively, execute the notebook with the repository runner:

```powershell
python tools/run_notebooks.py --only 00_download_annotations
```

The notebook's CLI command is:

```text
notebook annotations
```

but the repository runner is the documented replacement for normal execution.

**Outputs:** ZIP archive and four JSON annotation files under `data/raw/annotations`.

**VERIFY:**

```powershell
python -c "from pathlib import Path; p=Path('data/raw/annotations'); files=['captions_train2017.json','captions_val2017.json','instances_train2017.json','instances_val2017.json']; print([(f,(p/f).exists(),(p/f).stat().st_size if (p/f).exists() else 0) for f in files])"
```

All four paths must exist and have nonzero size.

**Dependencies:** Environment and configuration.

---

### 5. Create the dataset splits

**Purpose:** Build deterministic knowledge-base, validation, and test ID sets.

**Inputs:** The four annotation JSON files.

**Execution order encoded in `01_make_splits.ipynb`:**

1. Test: eligible COCO `val2017` images.
2. Validation: eligible COCO `train2017` images.
3. KB: eligible COCO `train2017` images excluding validation IDs.

Eligibility requires:

- at least five captions;
- at least one non-crowd object annotation;
- shorter side of at least 200 pixels.

**RUN:**

```powershell
jupyter lab
```

Execute `01_make_splits.ipynb`.

The repository runner equivalent is:

```powershell
python tools/run_notebooks.py --only 01_make_splits
```

**Seeds:**

| Split | Calculation | Seed |
|---|---|---:|
| Test | seed + 1 | 43 |
| Validation | seed + 2 | 44 |
| KB | seed + 3 | 45 |

The global seed is 42.

## Stratification

Validation and test sampling use:

### Complexity

Based on non-crowd COCO object count:

| Category | Object count |
|---|---:|
| simple | 1–2 |
| medium | 3–5 |
| complex | 6–9 |
| crowded | 10+ |

The target distribution is:

```text
25% simple, 30% medium, 30% complex, 15% crowded
```

### Greedy class coverage

The first step covers classes before filling complexity quotas:

1. Choose the rarest currently uncovered COCO category.
2. Among images containing it, choose the image covering the most other uncovered categories.
3. Use the seeded random generator to break ties.

This consumes up to half of the 100-image evaluation sample, so it can affect exact final strata.

### Interaction coverage

An image counts as an interaction when:

- it contains a person and another class;
- it contains at least two person instances; or
- it contains an animal supercategory plus another supercategory.

The configured minimum is 30%.

### Context rotation

When filling complexity strata, the implementation rotates candidates in:

```text
indoor -> outdoor -> mixed
```

The final frozen observed context counts are:

| Split | Indoor | Outdoor | Mixed |
|---|---:|---:|---:|
| Validation | 35 | 35 | 30 |
| Test | 37 | 33 | 30 |

## Outputs

- `artifacts/splits/kb.json`
- `artifacts/splits/val.json`
- `artifacts/splits/test.json`
- `artifacts/splits/splits_meta.json`

**VERIFY:** Inspect `splits_meta.json`.

Expected:

- KB 10,000
- Validation 100
- Test 100
- Val/test complexity quotas of 25/30/30/15
- No split overlap
- 80 classes covered in each evaluation split
- Validation interaction fraction at least 30%

**Dependencies:** Annotation download.

---

### 6. Download the selected images

**Purpose:** Download only the approximately 10,200 image records used by the frozen splits rather than all COCO images.

**Inputs:** The three split JSON files.

**RUN:**

```powershell
jupyter lab
```

Execute `01b_download_images.ipynb` after split creation.

The runner is:

```powershell
python tools/run_notebooks.py --only 01b_download_images
```

`01b_download_images.ipynb` reads each split record and downloads `coco_url`.

**Outputs:** Selected COCO images under:

```text
data/raw/images/train2017/
data/raw/images/val2017/
```

The frozen expected image count is:

```text
10,000 KB + 100 validation + 100 test = 10,200
```

**VERIFY:** Compare successfully downloaded files with split records. The image audit notebook writes the later authoritative result:

- missing file count should be zero in the frozen release;
- invalid IDs should be recorded in `artifacts/audit/invalid_images.json`.

**Dependencies:** Split files from Step 5.

---

### 7. Audit and preprocess images

**Purpose:** Find invalid files and create the normalized RGB release images.

**Inputs:**

- split JSON files;
- raw images;
- `config.yaml`.

**RUN:** Execute `02_preprocess_images.ipynb`.

```powershell
python tools/run_notebooks.py --only 02_preprocess_images
```

## Implementation steps

### Missing files

For each split record, `process()` checks whether the source path exists. A missing file receives:

```text
status = missing
```

### Corrupt files

Pillow is used to:

```python
Image.open(...).verify()
Image.open(...).load()
```

`UnidentifiedImageError`, `OSError`, `SyntaxError`, and `ValueError` become:

```text
status = corrupt
```

Corrupt file rows are written to the audit.

### EXIF orientation

The implementation reads the EXIF orientation tag and applies:

```python
ImageOps.exif_transpose(im)
```

### RGB conversion

The result is converted with:

```python
fixed.convert("RGB")
```

The frozen audit reports 24 grayscale images converted to RGB.

### Maximum-side resize

The maximum side is 512:

```text
scale = 512 / maximum(width, height)
```

Both axes are multiplied by the same scale, preserving aspect ratio. Pillow uses `Image.LANCZOS`.

Images are saved as JPEG:

```text
quality = 95
subsampling = 0
```

Encoder-specific model preprocessing occurs later and not in this notebook.

## Pixel statistics

The audit computes:

- brightness;
- contrast;
- colorfulness;
- sharpness;
- night/dark flag;
- named dominant colors.

## Exact duplicate detection

The notebook calls `common.sha256_file()` for raw source bytes. It groups rows having identical SHA-256 values.

## dHash detection

`visual.dhash()`:

1. converts the image to grayscale;
2. resizes to `(size + 1, size)`;
3. compares adjacent horizontal pixels;
4. records 64 bits for the default size 8.

Two hashes whose Hamming distance is 4 or less are candidate near-duplicates.

The notebook compares every val/test image with other val/test and KB images.

## Outputs

- normalized images under `artifacts/images_rgb/<split>/`
- `artifacts/tables/images.csv`
- `artifacts/audit/image_audit.json`
- `artifacts/audit/invalid_images.json`

**Expected frozen audit values:**

```text
10200 images
10200 valid
0 invalid/corrupt
0 EXIF corrections
24 RGB conversions
0 exact duplicates
0 dHash near-duplicates between val/test and other images
```

**VERIFY:** Inspect:

- `artifacts/audit/image_audit.json`
- `artifacts/audit/invalid_images.json`
- the first rows in `artifacts/tables/images.csv`

**Dependencies:** Split creation and image download.

---

### 8. Handle invalid and duplicate records

## Missing or corrupt images

If a rerun produces records in `invalid_images.json`, execute the split notebook again with exclusion enabled. `01_make_splits.ipynb` supports the `--exclude` argument, but there is no repository `01_make_splits.py` executable. In its notebook execution cell, pass the audit path by setting arguments before calling `main()`:

```python
import sys
sys.argv = ["01_make_splits.ipynb",
            "--exclude",
            "artifacts/audit/invalid_images.json"]
main()
```

`main()` reads that JSON and removes invalid IDs by assigning:

```python
bad = set(read_json(...).get("ids", []))
```

## Exact duplicate handling

For a group containing a KB image and evaluation image, `02b_dedup_repair.ipynb` removes KB candidates.

For exact duplicates where both copies are in the KB, it keeps the first and removes the remaining KB copies.

Evaluation images and evaluation-vs-evaluation conflicts can require manual treatment; `find_conflicts()` writes warning records for them.

## Near-duplicate handling

The repair notebook removes a matching KB ID when:

```text
dHash(match_eval, KB_candidate) <= configured threshold
```

It replaces each removed KB image with a randomly selected eligible train image.

Replacement is shuffled deterministically with:

```text
seed + 4 + repair-round-number
```

## Recorded repair

`artifacts/audit/dedup_repair.json` records:

| Removed KB image | Reason |
|---:|---|
| 186275 | Exact SHA-256 copy of a val/test image |
| 463132 | dHash near-duplicate of a val/test image |

Replacement IDs:

```text
149053
474404
```

The JSON does not record a one-to-one pairing between each removed image and each replacement.

**RUN:**

```powershell
jupyter lab
```

Execute `02b_dedup_repair.ipynb`.

If repairs are produced, repeat:

```text
01b_download_images
02_preprocess_images
02b_dedup_repair
```

until there is nothing to repair.

**IMPORTANT:** `tools/run_notebooks.py` does not include `02b_dedup_repair` in its `ORDER` list. It also does not include `03b_detect_objects`. Run either notebook manually or add deliberate stage execution only in a separate reproduction copy.

**VERIFY:** `dedup_repair.json` and:

```text
image_audit.json exact_duplicates = []
image_audit.json near_duplicates_eval_vs_others = []
```

**Dependencies:** Image preprocessing audit.

---

### 9. Build captions and metadata

**Purpose:** Create raw and cleaned KB captions, raw evaluation references, and image metadata.

**RUN:**

```powershell
jupyter lab
```

Execute `03_captions_metadata.ipynb`.

Or:

```powershell
python tools/run_notebooks.py --only 03_captions_metadata
```

## Caption selection

For each image, captions are:

- grouped by `image_id`;
- sorted by caption ID;
- limited to the configured first five.

## KB caption table

`artifacts/tables/kb_captions.jsonl` contains:

```json
{
  "caption_id": 755795,
  "image_id": 78,
  "caption_raw": "A lighted owl candle sits next to a clock.",
  "caption_clean": "a lighted owl candle sits next to a clock",
  "n_tokens": 9
}
```

## Evaluation references

These files contain raw text only:

```text
eval_refs_val.jsonl
eval_refs_test.jsonl
```

Evaluation references must not be copied into `kb_captions.jsonl` and must not be text-indexed.

## Caption cleaning

`common.clean_caption()`:

1. strips whitespace;
2. lowercases;
3. replaces unsupported punctuation;
4. handles apostrophes surrounded by spaces;
5. collapses repeated whitespace.

`tokenize()` cleans text again and splits on whitespace.

## Metadata

The metadata JSON uses COCO object instances:

- category ID and name;
- supercategory;
- normalized bounding box;
- area fraction;
- object size;
- object dominant pixel color;
- geometric relations;
- scene context;
- complexity;
- day/night state;
- caption-derived attributes for KB only.

Object metadata for val/test comes from COCO ground truth and is therefore an **oracle**, not realistic predicted metadata.

## Optional detector metadata

**OPTIONAL:**

Execute `03b_detect_objects.ipynb` only to generate predicted detector metadata for query images.

`config.yaml` uses:

```text
facebook/detr-resnet-50
confidence threshold 0.7
```

It is not required for the core frozen 10,000-image KB metadata and is not included in the full runner's default order.

**Recorded run** (`artifacts/audit/detector_info.json`):

- 200 images processed (`n_images` = 200: the historical validation and test images).
- 1,568 detections in `artifacts/tables/detections_detr.jsonl`.

**Not recorded / not evaluated:**

- The `transformers` version and the model revision are not recorded.
- Detector accuracy (IoU, mAP) was not evaluated.
- The frozen validation/test metadata remain the COCO annotations (oracle).

**Outputs:**

- `artifacts/tables/kb_captions.jsonl`
- `artifacts/tables/eval_refs_val.jsonl`
- `artifacts/tables/eval_refs_test.jsonl`
- `artifacts/tables/metadata.jsonl`
- `artifacts/audit/metadata_info.json`
- optional `artifacts/tables/detections_detr.jsonl`

**VERIFY:**

```text
KB: 10,000 images x 5 captions = 50,000 rows
Val: 100 images x 5 captions = 500 rows
Test: 100 images x 5 captions = 500 rows
```

Check that `caption_raw` is not overwritten and that evaluation objects are available as documented oracle metadata.

**Dependencies:** Completed split repair and image preprocessing.

---

### 10. Run exploratory data analysis

**Purpose:** Generate 11 figures and summary statistics.

**RUN:**

```powershell
jupyter lab
```

Execute `05_eda.ipynb`.

Or:

```powershell
python tools/run_notebooks.py --only 05_eda
```

The notebook executes only after caption and metadata generation.

**Outputs:**

```text
artifacts/eda/01_top50_classes.png
artifacts/eda/02_scene_complexity.png
artifacts/eda/03_spatial_relations.png
artifacts/eda/04_caption_length_zipf.png
artifacts/eda/05_pos_distribution.png
artifacts/eda/06_cooccurrence_top30.png
artifacts/eda/07_objects_vs_caption_length.png
artifacts/eda/08_diversity_context_light_size.png
artifacts/eda/09_dominant_colors.png
artifacts/eda/10_sample_grid.png
artifacts/eda/11_attributs_couleur_forme.png
artifacts/eda/stats.json
artifacts/eda/EDA_REPORT.md
```

**VERIFY:** Compare:

- 10,000/100/100 image counts;
- 50,000/500/500 caption counts;
- average objects per image near 8.05 in the KB;
- mean KB caption length near 10.48 tokens;
- 80 classes present in all splits.

POS tags use NLTK or spaCy fallback logic, depending on available packages. The repository does not pin spaCy, but NLTK is installed.

**Dependencies:** Captions, metadata, and normalized images.

---

### 11. Generate embeddings

**Purpose:** Produce image vectors and matching text vectors for retrieval.

**RUN:**

```powershell
jupyter lab
```

Execute `04_embeddings_index.ipynb`.

Or:

```powershell
python tools/run_notebooks.py --only 04_embeddings_index
```

## `HFEncoder`

For CLIP and SigLIP:

```text
AutoProcessor
AutoTokenizer
get_image_features()
get_text_features()
```

For DINOv2:

```text
AutoImageProcessor
model(...).last_hidden_state[:, 0]
```

## Model details

| Encoder key | Model ID | Image/text | Dimension |
|---|---|---|---:|
| `clip-b32` | `openai/clip-vit-base-patch32` | image + text | 512 |
| `siglip-b16` | `google/siglip-base-patch16-224` | image + text | 768 |
| `dinov2-s` | `facebook/dinov2-small` | image only | 384 |
| SBERT | `sentence-transformers/all-MiniLM-L6-v2` | text only | 384 |

## Normalization and caching

`04_embeddings_index.ipynb` uses `l2n()`:

```python
x / max(norm(x), 1e-12)
```

The output arrays are float32 and L2-normalized.

`cached_encode()`:

- aligns cache rows with ID arrays;
- reuses complete matching caches;
- writes missing rows atomically;
- recomputes only when requested.

## SigLIP-only reproduction

The repository also documents an offline v1.1 extraction:

```powershell
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe tools\run_siglip_only.py
```

The script:

- executes required setup and definition cells from `04_embeddings_index.ipynb`;
- sets the notebook arguments to:

```text
--encoders siglip-b16 --skip-sbert
```

- runs `main()` once.

The initial SigLIP extraction was performed under:

```text
Python 3.13.3
torch 2.15.0+cpu
transformers 5.22.1
faiss 1.15.1
numpy 2.5.4
```

The frozen manifest records Python 3.13.7.

## Outputs

For each encoder:

```text
artifacts/embeddings/<encoder>/ids_kb.npy
artifacts/embeddings/<encoder>/ids_val.npy
artifacts/embeddings/<encoder>/ids_test.npy
artifacts/embeddings/<encoder>/kb_img.npy
artifacts/embeddings/<encoder>/val_img.npy
artifacts/embeddings/<encoder>/test_img.npy
```

CLIP and SigLIP also generate:

```text
kb_txt.npy
kb_txt_caption_ids.npy
kb_txt_image_ids.npy
```

SBERT:

```text
artifacts/embeddings/sbert_kb_txt.npy
artifacts/embeddings/sbert_kb_txt_caption_ids.npy
```

**VERIFY:**

```text
CLIP image arrays: 10000/100/100 x 512
SigLIP image arrays: 10000/100/100 x 768
DINOv2 image arrays: 10000/100/100 x 384
CLIP and SigLIP text arrays: 50000 x their dimensions
```

Verify vector norms are approximately 1 and that every ID array length matches its vector array.

**Dependencies:** 10,200 normalized images and split caption tables.

---

### 12. Build FAISS indexes

FAISS is a vector index used for nearest-neighbour retrieval. The notebook builds indexes only after all embeddings for an encoder exist.

## Flat index

```python
faiss.IndexFlatIP(dimension)
```

Because vectors are L2-normalized:

```text
inner product = cosine similarity
```

## HNSW index

```python
faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
```

Configuration:

```text
M = 32
efConstruction = 200
efSearch = 64
```

HNSW is approximate; it is compared against the exact index on the test queries.

## Text index

CLIP and SigLIP text captions use:

```text
<encoder>_txt_flatip.faiss
```

## Outputs

For each:

```text
artifacts/index/clip-b32_flatip.faiss
artifacts/index/clip-b32_hnsw.faiss
artifacts/index/clip-b32_txt_flatip.faiss

artifacts/index/siglip-b16_flatip.faiss
artifacts/index/siglip-b16_hnsw.faiss
artifacts/index/siglip-b16_txt_flatip.faiss

artifacts/index/dinov2-s_flatip.faiss
artifacts/index/dinov2-s_hnsw.faiss
```

The report:

```text
artifacts/index/index_report.json
```

**Expected index sizes:** 10,000 vectors.

| Encoder | Dimension | Flat MB | HNSW MB |
|---|---:|---:|---:|
| CLIP | 512 | 20.480 | 23.198 |
| SigLIP | 768 | 30.720 | 33.438 |
| DINOv2 | 384 | 15.360 | 18.078 |

**Expected HNSW Recall@10:**

| Encoder | Recall@10 |
|---|---:|
| CLIP | 1.000 |
| SigLIP | 0.999 |
| DINOv2 | 0.997 |

**VERIFY:** Load the exact index and confirm `ntotal == 10000`. Then verify first-vector retrieval against its own ID as implemented in QC.

**Dependencies:** Embeddings and KB IDs.

---

### 13. Compare encoders and select the primary configuration

**Purpose:** Choose one configuration using validation data without consulting test selection.

**RUN:**

```powershell
jupyter lab
```

Execute `06_select_config.ipynb`.

Alternatively:

```powershell
python tools/run_notebooks.py --only 06_select_config
```

## Configurations compared

The implementation evaluates:

```text
clip-b32 | image-to-image
clip-b32 | image-to-text
dinov2-s | image-to-image
siglip-b16 | image-to-image
siglip-b16 | image-to-text
```

## Image-to-image

1. Search KB image vectors.
2. Retrieve a KB image.
3. Use the first caption of the retrieved image as prediction.

## Image-to-text

1. Search the KB caption vectors.
2. Take the retrieved caption text directly.

## Evaluation

The notebook:

- computes per-image CIDEr;
- creates a 10-fold split of per-image results;
- creates 2,000 seeded bootstrap samples;
- calculates 95% percentile intervals;
- uses Wilcoxon paired signed-rank comparisons against the best;
- calculates top-one Jaccard class overlap.

## Validation selection results

| Configuration | CIDEr |
|---|---:|
| SigLIP image-to-text | 0.7222567867 |
| SigLIP image-to-image | 0.6735370594 |
| CLIP image-to-text | 0.5584101765 |
| DINOv2 image-to-image | 0.5541039133 |
| CLIP image-to-image | 0.5285707245 |

The recommendation in the frozen file is:

```text
siglip-b16|img2txt
```

The released primary encoder is therefore `siglip-b16`.

Test results are report-only and are not used to select the encoder.

## v1.0 difference

Before Sarah added SigLIP, the backup selection selected:

```text
clip-b32|img2txt
```

`config_v1.0_backup.yaml` also sets:

```text
primary_encoder: clip-b32
```

**Output:**

```text
artifacts/audit/preproc_config_selection.json
reports/tables/encoder_comparison_val.csv
reports/tables/encoder_comparison_test.csv
reports/figures/encoder_comparison_val.png
```

**VERIFY:**

- selected configuration is `siglip-b16|img2txt`;
- selection report was generated from validation only;
- test is marked report-only in `encoder_comparison_test.csv`;
- SigLIP has dimension 768 in embeddings and FAISS indexes.

**Dependencies:** Embeddings and FAISS indexes.

---

### 14. Run quality control and leakage audit

**Purpose:** Verify split integrity, captions, metadata, embeddings, indexes, retrieval behavior, and smoke-test retrieval quality.

**RUN:**

```powershell
jupyter lab
```

Execute `07_qc_leakage.ipynb`.

The repository runner is:

```powershell
.\.venv\Scripts\python.exe tools\run_qc_only.py
```

The runner executes selected setup/definition cells in `07_qc_leakage.ipynb`, then calls `main()`.

## The 40 checks

The exact frozen check names are:

| Checks | Verification |
|---|---|
| 1–2 | KB size 10,000 and ID list hash against `splits_meta.json`. |
| 3–4 | Validation size 100 and ID list hash. |
| 5–6 | Test size 100 and ID list hash. |
| 7 | Pairwise split disjointness. |
| 8 | No corrupt or missing image statuses. |
| 9 | Image IDs are unique in `images.csv`. |
| 10 | No exact SHA-256 duplicates within or between splits. |
| 11 | No pixel-stat NaNs. |
| 12 | Five KB captions for every KB image. |
| 13 | Five references for every validation image. |
| 14 | Five references for every test image. |
| 15 | KB caption IDs are unique. |
| 16 | No validation/test image ID appears in KB captions. |
| 17 | No evaluation caption ID appears in the KB. |
| 18 | Raw captions remain non-empty and raw text may differ from cleaned text. |
| 19 | Validation normalized caption overlap is reported as information, not failure. |
| 20 | Test normalized caption overlap is reported as information, not failure. |
| 21 | Metadata exists for every image. |
| 22 | Text-derived attributes occur only in KB metadata. |
| 23–25 | CLIP KB, validation, and test embedding shapes are `(10000, 512)`, `(100, 512)`, and `(100, 512)`. |
| 26 | CLIP flat index has the KB size and supports self-retrieval. |
| 27–29 | SigLIP KB, validation, and test embedding shapes are `(10000, 768)`, `(100, 768)`, and `(100, 768)`. |
| 30 | SigLIP flat index has the KB size and supports self-retrieval. |
| 31–33 | DINOv2 KB, validation, and test embedding shapes are `(10000, 384)`, `(100, 384)`, and `(100, 384)`. |
| 34 | DINOv2 flat index has the KB size and supports self-retrieval. |
| 35 | Validation query IDs are never returned as themselves in the top 10. |
| 36 | SigLIP cosine and dHash validation near-duplicate counts against KB are reported. |
| 37 | Test query IDs are never returned as themselves in the top 10. |
| 38 | SigLIP cosine and dHash test near-duplicate counts against KB are reported. |
| 39 | Validation nearest-neighbour caption CIDEr exceeds the random caption baseline. |
| 40 | Test nearest-neighbour caption CIDEr exceeds the random caption baseline. |

The exact names and status details are in:

```text
reports/leakage_audit.md
artifacts/qc/qc_report.md
artifacts/qc/qc_report.json
```

## Anti-leakage checks

The implementation checks:

- split image-ID disjointness;
- KB captions contain no val/test image IDs;
- evaluation caption IDs do not appear in KB caption IDs;
- query images cannot retrieve the same image ID in `Release.search_images()`;
- cross-split dHash candidates;
- SigLIP cosine score candidates at or above 0.97.

## Caption overlap

The frozen report records:

```text
validation cleaned caption overlap = 1.0%
test cleaned caption overlap = 1.4%
```

This is not automatically leakage because:

- the strings are generic repeated COCO phrases;
- the evaluation and KB image IDs are different;
- the comparison does not establish visual or file identity.

**Outputs:**

- `artifacts/qc/qc_report.json`
- `artifacts/qc/qc_report.md`
- `reports/leakage_audit.md`

**Expected result:**

```text
40 checks
40 PASS
0 FAIL
0 WARN
```

Check 40 count is the frozen v1.1 result.

**Dependencies:** Complete embeddings, captions, metadata, indexes, splits, and audit.

---

### 15. Demonstrate retrieval without changing artifacts

**Purpose:** Verify the public release-loading API and top-5 retrieval.

Notebook 09 is read-oriented, but inspect it before running against a frozen release.

It demonstrates:

```python
R = Release("artifacts", cfg["primary_encoder"])
Q = R.image_embeddings("test")
ids = R.ids("test")
S, I = R.search_images(Q, k=5, query_ids=ids)
```

It also retrieves raw associated KB captions and evaluation references separately.

**RUN only in an intentional reproduction copy:**

```powershell
jupyter lab
```

Execute `09_demo_retrieval.ipynb`.

**VERIFY:**

- no self-match assertion is raised;
- each result image belongs to the KB split;
- captions come from `kb_captions_of()`;
- evaluation references come only from `eval_refs("test")`.

**Dependencies:** Frozen/rebuilt release files.

---

### 16. Freeze and verify the release

**Purpose:** Create a manifest and make release files read-only.

Notebook 08 refuses to freeze unless QC passed.

**RUN the frozen document workflow:**

```powershell
.\.venv\Scripts\python.exe tools\run_freeze_only.py
.\.venv\Scripts\python.exe tools\run_freeze_only.py --verify
```

`run_freeze_only.py` executes notebook cells 1, 3, and 5 and calls `main()`.

For a normal notebook run, execute `08_freeze.ipynb`.

The freeze notebook:

1. checks for QC;
2. refuses freeze if QC has failures;
3. hashes each artifact file except the manifest;
4. writes `artifacts/manifest.json`;
5. sets artifact files to read-only permissions;
6. optionally creates `release_g1-v1.1.tar.gz` without JPEG images.

**Outputs:**

- `artifacts/manifest.json`
- read-only artifact files
- optional lightweight archive.

The manifest contains:

- release version;
- seed;
- sizes;
- primary encoder;
- encoder list;
- environment versions;
- per-file SHA-256;
- byte size.

**VERIFY with the API:**

```powershell
python -c "from rag_data import Release; bad=Release('artifacts').verify(); print('release OK' if not bad else bad)"
```

Expected:

```text
release OK
```

**Dependencies:** 40-check QC pass and all artifact generation.

---

## Full runner behavior

### Documented full sequence

The README lists:

```text
00_download_annotations
01_make_splits
01b_download_images
02_preprocess_images
02b_dedup_repair
03_captions_metadata
04_embeddings_index
05_eda
06_select_config
07_qc_leakage
08_freeze
09_demo_retrieval
```

It states that `02b` is rerun as necessary until it reports no repairs.

### Actual `tools/run_notebooks.py` default order

The script’s `ORDER` currently contains:

```text
00_download_annotations
01_make_splits
01b_download_images
02_preprocess_images
03_captions_metadata
04_embeddings_index
05_eda
06_select_config
07_qc_leakage
08_freeze
09_demo_retrieval
```

It omits:

- `02b_dedup_repair`
- `03b_detect_objects`

Run those manually when needed.

### Normal full runner

```powershell
python tools/run_notebooks.py
```

It:

- uses the environment variable `RAG_CONFIG`;
- writes executed copies under `executed_notebooks/`;
- unlocks existing artifact directories before reproduction;
- stops if an executed notebook raises a cell error.

### Synthetic offline test

```powershell
python tools/make_fake_coco.py
python tools/run_notebooks.py --config config_test.yaml --skip 00_download_annotations 01b_download_images
```

This creates a small synthetic environment intended to test pipeline plumbing without the real dataset.

It does not reproduce the frozen 10,000-image release.

---

# A. Prerequisites

## Software

- Git
- Python 3.13.x
- PowerShell or a compatible shell
- JupyterLab
- CPU is sufficient for the delivered pipeline, although the official run of the current release used CPU.
- COCO images require network access during image download unless raw files are already available.

## Packages

The exact pinned package list is in `requirements.txt`, including:

- NumPy
- pandas
- SciPy
- PyYAML
- Pillow
- Matplotlib
- FAISS CPU
- pycocoevalcap
- pycocotools
- NLTK
- PyTorch
- Transformers
- sentence-transformers
- SentencePiece
- protobuf
- ImageHash
- JupyterLab and notebook support

## Dataset files

Required raw annotation files:

```text
captions_train2017.json
captions_val2017.json
instances_train2017.json
instances_val2017.json
```

The reproduction should ultimately contain records for:

```text
10,000 KB images
100 validation images
100 test images
```

## Storage and hardware

The repository does not document a precise disk requirement. It does document that the full 18 GB COCO train image archive should not be downloaded: selected split images are enough.

The notebook and runners support CPU; embedding generation may take a long time. The frozen SigLIP report records roughly:

```text
3020 seconds for image embeddings
3774 seconds for text embeddings
```

These are measurements on the recorded run and are not a general requirement.

---

# B. Reproduction checklist

- [ ] Created a separate working copy.
- [ ] Confirmed `sarah-phase1`.
- [ ] Created Python 3.13 environment.
- [ ] Installed `requirements.txt`.
- [ ] Downloaded or placed the four COCO annotation JSON files.
- [ ] Generated KB, validation, and test splits.
- [ ] Verified 10,000/100/100 split sizes.
- [ ] Downloaded the selected 10,200 images.
- [ ] Ran image auditing.
- [ ] Confirmed no missing/corrupt images.
- [ ] Confirmed no exact duplicate files.
- [ ] Ran dHash near-duplicate screening.
- [ ] Ran `02b_dedup_repair`.
- [ ] Repeated download/preprocess repair until no repair remained.
- [ ] Generated five captions for every KB image.
- [ ] Generated five references for every evaluation image.
- [ ] Generated metadata.
- [ ] Ran EDA.
- [ ] Generated CLIP, DINOv2, SigLIP, and optional SBERT embeddings.
- [ ] Built flat, HNSW, and text FAISS indexes.
- [ ] Selected SigLIP image-to-text on validation.
- [ ] Did not use the test split for selection.
- [ ] Ran all 40 QC checks.
- [ ] Verified anti-leakage behavior.
- [ ] Frozen the release and verified the manifest.

---

# C. Expected final outputs

## Splits and tables

- `artifacts/splits/kb.json`
- `artifacts/splits/val.json`
- `artifacts/splits/test.json`
- `artifacts/splits/splits_meta.json`
- `artifacts/tables/images.csv`
- `artifacts/tables/kb_captions.jsonl`
- `artifacts/tables/eval_refs_val.jsonl`
- `artifacts/tables/eval_refs_test.jsonl`
- `artifacts/tables/metadata.jsonl`

## Audits and repair

- `artifacts/audit/image_audit.json`
- `artifacts/audit/invalid_images.json`
- `artifacts/audit/dedup_repair.json`
- `artifacts/audit/preproc_config_selection.json`

## Embeddings and indexes

- `artifacts/embeddings/clip-b32/*`
- `artifacts/embeddings/siglip-b16/*`
- `artifacts/embeddings/dinov2-s/*`
- `artifacts/embeddings/sbert_kb_txt.npy`
- `artifacts/index/*_flatip.faiss`
- `artifacts/index/*_hnsw.faiss`
- `artifacts/index/*_txt_flatip.faiss`
- `artifacts/index/index_report.json`

## EDA and QC

- all 11 files in `artifacts/eda/`
- `artifacts/eda/stats.json`
- `artifacts/eda/EDA_REPORT.md`
- `artifacts/qc/qc_report.json`
- `artifacts/qc/qc_report.md`
- `reports/leakage_audit.md`

## Finalization

- `artifacts/manifest.json`
- read-only artifact files
- optional `release_g1-v1.1.tar.gz`

---

# D. Success criteria

A reproduction matches the repository when all of the following are true:

1. Split sizes are 10,000, 100, and 100.
2. The KB contains 50,000 captions.
3. Validation and test contain 500 raw references each.
4. There are no invalid, missing, or known exact duplicate source images.
5. The frozen duplicate repairs can be explained through `dedup_repair.json`.
6. SigLIP image vectors have dimension 768.
7. SigLIP text vectors have dimension 768.
8. SigLIP is selected on validation:

```text
siglip-b16 | img2txt
```

9. The test split is report-only.
10. FAISS contains 10,000 vectors for each image KB index.
11. CLIP and SigLIP also have 50,000-vector text indexes.
12. QC is:

```text
40 PASS, 0 FAIL, 0 WARN
```

13. A query image cannot retrieve itself.
14. `Release.verify()` reports no missing or changed files.

Exact numeric reproduction may vary in timing fields and environment-dependent computations. `index_report.json` explicitly contains timings, and the README notes that reruns can differ in those timing fields.

---

# E. Troubleshooting

## Missing images

The repository reports downloading failures and asks for the image step to be rerun; `01b_download_images.ipynb` contains retry and recovery behavior.

Do not continue with a corrupted subset: rerun image download and inspect `invalid_images.json`.

## Corrupt images

The preprocessing notebook writes a `corrupt` status. After correction, rerun the relevant split and preprocessing stages in a separate copy.

## Duplicate repair required

Run:

```text
01b_download_images
02_preprocess_images
02b_dedup_repair
```

until the repair notebook says nothing needs repairing.

## Artifact writes blocked by read-only permissions

This is expected after `08_freeze.ipynb`. Use a separate working copy. The repository runner deliberately unlocks artifacts before execution, but frozen-artifact safety means this should not be done in the original release directory.

## SigLIP extraction issues

The repository provides:

- a separate offline runner;
- cached SigLIP extraction;
- a guard that aborts unless exactly `siglip-b16 --skip-sbert` is selected.

Models are loaded from Hugging Face and cached locally. If the model is not cached, network access or a separately populated Hugging Face cache is required.

## `03b_detect_objects` is optional

It supplies detector metadata for realistic query metadata. It is not required for the standard frozen oracle-metadata release.

## `02b_dedup_repair` is absent from the default runner

The README gives manual execution guidance and notes repeated download/preprocess repair. The current `tools/run_notebooks.py` `ORDER` omits `02b`. Execute it manually.

## Text overlap appears

The repository itself treats its 1.0% validation and 1.4% test cleaned text overlap as informative rather than direct evidence of leakage.

Do not convert this into an automatic failure unless a future audit also finds matching image IDs, exact hashes, or retrieval self-matches.

---
# F. Glossary

## SHA-256

A cryptographic digest. Equal SHA-256 values mean the repository treats raw source bytes as identical, within normal hash assumptions.

## dHash

A perceptual difference hash calculated from adjacent grayscale brightness comparisons. A Hamming distance of 4 or less is treated as near-identical in the repository.

## Embedding

A numeric vector representing an image or text. Similar vectors are intended to represent related content.

## CLIP

A vision-language model trained to connect image and text representations. The release uses CLIP ViT-B/32 vectors of dimension 512.

## DINOv2

A self-supervised vision encoder that produces image representations. The implementation uses the CLS token; dimension is 384.

## SigLIP

A vision-language encoder based on sigmoid image-text training. It produced the selected image-to-text configuration and has dimension 768.

## Sentence-BERT

A sentence/text embedding family. The repository uses `all-MiniLM-L6-v2` for cleaned KB captions, with dimension 384.

## FAISS

A library for storing vectors and searching for nearest neighbours.

## Flat index

An exact index that compares a query against every vector in the KB.

## HNSW

Hierarchical Navigable Small World: an approximate graph search method that is usually faster than exact search but not guaranteed to retrieve every exact neighbour.

## Recall@10

The fraction of exact top-10 neighbours found by HNSW.

## Data leakage

Evaluation information entering training, selection, metadata, or retrieval. The repository prevents this through split disjointness, separate evaluation captions, duplicate repair, embedding audits, and retrieval self-match checks.

## Stratification

Choosing samples so that their complexity, context, interaction, and class coverage reflect project-defined quotas rather than an entirely random distribution.


