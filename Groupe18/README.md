# A Comparative Study of RAG Techniques for Generating Image Captions

## Project objective

This repository prepares and audits a reproducible COCO image-captioning retrieval corpus for a comparative study of retrieval-augmented caption generation. The current Phase 1 implementation freezes the image splits, preserves reference captions, records object and visual metadata, extracts CLIP image and text embeddings, and evaluates train-gallery retrieval. It does **not** yet implement caption generation or compare RAG generation methods; those are Phase 2 work.

## Dataset and frozen splits

The source is MS COCO 2017 train and validation images, instance annotations, and captions. The project selected 10,000 train images from `train2017` and 100 validation plus 1,000 test images from `val2017`. Each selected image has five reference captions, for 50,000 / 500 / 5,000 records respectively. Split manifests are frozen in `data/selected/{train,validation,test}_ids.json` and matching CSVs. Do not run `select` or `preprocess` on this completed checkout.

| Split | Images | Reference captions | Processed images |
|---|---:|---:|---:|
| Train | 10,000 | 50,000 | 10,000 |
| Validation | 100 | 500 | 100 |
| Test | 1,000 | 5,000 | 1,000 |
| Total | 11,100 | 55,500 | 11,100 |

The assignment audit documents a conflict between a 100-image and a 1,000-image test requirement. The existing 1,000-image test split is retained pending clarification. See the Phase 1 compliance report; overall assignment compliance remains **PARTIAL**.

## Repository map

```text
configs/config.json                 Paths-independent pipeline and model settings
data/raw/                           Original COCO images and annotation JSON files
data/selected/                      Frozen split manifests and replacement history
data/processed/{train,validation,test}/
                                    224 x 224 RGB letterboxed images
data/captions/                       Selected raw/cleaned caption records and POS output
data/metadata/                       Authoritative per-split image JSON and supporting analysis
src/                                 CLI stages and analysis modules
embeddings/                          Saved image and text vector arrays and mappings
vector_database/                     Train-only FAISS image and caption indexes
reports/                             QC, compliance, analysis, retrieval, and stage reports
visualizations/                      Figures produced by analysis stages
main.py                              Single command-line entry point
requirements.txt                     Python dependencies
```

The authoritative image metadata is `data/metadata/train.json`, `validation.json`, and `test.json`. Supporting sources include `metadata_schema.json`, the metadata quality report, color/shape summaries, per-image geometric summaries, and CLIP prediction records. The split JSONs distinguish COCO annotations, measured image pixels, model predictions, and box-derived geometry through their field names and provenance. Category names are included alongside COCO IDs. Root-level split JSONs remain authoritative; no `data/metadata/{train,validation,test}/` export directories are used.

## Methods and saved outputs

### Selection and preprocessing

Selection used seed `20261106` and a rarity-first multilabel greedy procedure. Training IDs come from `train2017`; validation and test IDs come from disjoint selections in `val2017`. Preprocessing applies EXIF orientation, RGB conversion, aspect-preserving resize, and centered black letterboxing to 224 x 224. CLIP normalization is applied when encoding, not written into processed JPEGs. Raw images and captions are read-only inputs for the analysis stages.

### Caption and image analysis

Caption vocabulary analysis uses Unicode NFKC normalization and case folding, then tokenizes Unicode letters/numbers while retaining internal straight or curly apostrophes. It does not stem or lemmatize. Frequency and length statistics are in `reports/statistical_analysis/`, with figures in `visualizations/`.

Color summaries use coarse HSV color families and luminance/saturation measurements on original images. Edge density is a global contour/texture proxy, not object-shape recognition. Spatial relations are computed geometrically from COCO boxes (ordering, containment, overlap, and proximity). They are inferences, not COCO semantic relationship annotations and do not establish actions, contact, or support. The validation preprocessing comparison tests letterbox, stretch, and center crop on the existing 100 validation images using geometry-based criteria; it is not a historical experiment or a downstream model performance result.

COCO object labels and boxes remain annotation data. CLIP zero-shot scene/context predictions are separate model outputs with relative prompt scores, not calibrated probabilities. Pixel-level colors are not attributed to detected objects.

### Embeddings and retrieval

Image and caption vectors use `openai/clip-vit-base-patch32` (512 dimensions, float32, L2 normalized). Image arrays cover the three frozen splits; text vectors cover only the 50,000 raw training captions. Separate FAISS `IndexFlatIP` indexes contain the 10,000 training images and 50,000 training captions. Because vectors are normalized, inner product is cosine similarity. Validation/test image queries search a training-only image gallery. Existing QC records 20 successful held-out queries at top-5, zero self matches, and mean top-1 cosine 0.8260. Similarity retrieval is a Phase 1 component, not a generated-caption evaluation.

## Quality and compliance

`reports/final_quality_control.json` currently reports **PASS** for pipeline checks, including split counts, five captions per image, processed-image decoding, duplicate/leakage checks, embeddings and mappings, FAISS readback, and retrieval. This does not mean the assignment is fully satisfied. `reports/phase1_final_compliance.json` and `reports/PHASE1_FINAL_COMPLIANCE_REPORT.md` give the requirement-level result, which remains **PARTIAL** due to unresolved source-document and test-size questions and other documented reproducibility limits. Consult the reports for evidence and details.

## Installation

Use Python 3.10 or newer. Install the listed packages and install the spaCy English model once for POS tagging:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Model stages use locally cached model files. They do not download checkpoints implicitly; a missing local checkpoint is an explicit error.

## Commands

List available stages:

```powershell
python main.py --help
```

Run focused non-model analyses or validate the existing project:

```powershell
python main.py extended_analysis
python main.py metadata_redesign
python main.py quality
```

Other existing stages are `embeddings`, `text_embeddings`, `retrieval`, `caption_index`, `phash`, `pos`, `visual`, `replacement_plan`, and `replacement_apply`. `remaining` runs multiple stages, including potentially expensive model and index stages. Do not run it just to validate cleanup. `explore`, `select`, `preprocess`, and `all` are for a new dataset preparation, not for this frozen project.

## Reproducibility

The GitHub checkout omits the original and processed COCO image trees, large COCO annotation dumps, training embedding arrays, FAISS index binaries, and the oversized geometric detail CSV. These files exceed practical GitHub repository limits or can be regenerated. Download the COCO 2017 source data and run the documented pipeline stages to recreate local artifacts.

Keep `configs/config.json`, the selected ID manifests, source annotation files, and saved run metadata with the project. For a focused analysis, run its command from the repository root and inspect the corresponding report. `python main.py quality` refreshes pipeline QC and its final status reports; it does not rebuild embeddings or indexes. The project cleanup audit records SHA-256 hashes before and after cleanup for protected images, captions, manifests, embeddings, indexes, and existing reports in `reports/cleanup_report.json`.

## Known limitations

- Phase 1 does not train or evaluate a caption generator and does not yet compare RAG generation techniques.
- The original assignment PDF was not available to the compliance audit, and the test-size discrepancy remains unresolved.
- Historical EXIF/conversion counts and a unified environment snapshot are not available.
- Geometry-derived relations and pixel/edge statistics have the interpretation limits described above.
- CLIP prompt scores and POS tags are model outputs and can be incorrect.
