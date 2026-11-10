# Dataset and data handling

## Scope

This repository prepares and analyzes data for Phase 1 of the image-captioning RAG project. It does not generate final captions or evaluate a completed RAG system. The image retrieval check is a functional validation, not a relevance benchmark.

The dataset is Microsoft COCO 2017:

- Images and annotations: [COCO dataset](https://cocodataset.org/#download)
- Image encoder: `openai/clip-vit-base-patch32`
- Detailed report: [English report (PDF)](output/pdf/Phase1_report_EN.pdf) and [Markdown edition](output/pdf/Phase1_report_EN.md).

COCO images retain their original owners' and source-site licensing terms. Consult the COCO and image-level license information before reusing or redistributing images. The images are intentionally not committed to this repository.

## Frozen partitions

The ID lists in `artifacts/splits/` are the source of truth. Do not regenerate them with `splits --force`.

| Partition | COCO source | Images | Captions | Use |
|---|---|---:|---:|---|
| Knowledge base | `train2017` | 10,000 | 50,000 | The only partition used to build retrieval indexes |
| Validation | `train2017` | 100 | 500 | Development and retrieval functional validation |
| Test | `val2017` | 1,000 | 5,000 | Separate held-out analysis; never an index source |
| Comparison subset | `val2017` | 100 | 500 | The original test-100 sample, nested in the 1,000-image test |

There are 11,100 distinct images across the knowledge base, validation, and test partitions. The comparison subset is included in the test total and must not be added again. The test ID list is a locally frozen, provisional sample; no class-wide official ID list was supplied. See `artifacts/splits/split_manifest.json` for the seed, sampling method, sizes, and hashes.

## Source files and generated artifacts

The download commands place the official COCO annotation archives and selected JPEGs under:

```text
data/raw/
  annotations/
  train2017/
  val2017/
```

Generated metadata is stored as JSON Lines in `artifacts/metadata/`; split IDs and their manifest are in `artifacts/splits/`. EDA outputs are in `reports/`. Current retrieval validation summaries and neighbor lists are in [`results/retrieval/validation/`](results/retrieval/validation/).

COCO images, downloaded annotations, model weights, intermediate caches, and vector embeddings are local inputs or derived data. They are not required to review the code and summary results on GitHub, and should not be added to a submission archive.

## Processing and leakage controls

- Images are opened with Pillow, EXIF orientation is applied, and RGB images are resized and center-cropped to `224 x 224` for CLIP.
- Original COCO captions are preserved. A separate clean form normalizes whitespace only.
- Objects and bounding boxes come from COCO instance annotations. Caption-derived attributes are explicitly heuristic and are not visual detector predictions.
- Image IDs in the knowledge base and validation split are disjoint. The test set comes from `val2017` and is kept out of retrieval indexing.
- The frozen split manifest records the configuration hash, source annotation hashes, seed (`42`), split sizes, and test-subset relationship.

## Reproduce the local data pipeline

From the repository root on Windows, install Python 3.12 and the packages in `requirements.txt`, then run:

```powershell
py -3.12 -m venv imp
.\imp\Scripts\python.exe -m pip install -r requirements.txt
.\imp\Scripts\python.exe -m src.cli download-annotations
.\imp\Scripts\python.exe -m src.cli download-images
.\imp\Scripts\python.exe -m src.cli download-nlp
.\imp\Scripts\python.exe -m src.cli download-model
.\imp\Scripts\python.exe -m src.cli metadata
.\imp\Scripts\python.exe -m src.cli eda
.\imp\Scripts\python.exe -m src.cli --offline embeddings
.\imp\Scripts\python.exe -m src.cli index
.\imp\Scripts\python.exe -m src.cli --offline validate-retrieval --top-k 5
```

The first four commands download data or model resources. The embedding and index steps require those local inputs and can take time on CPU. For the full sequence and caveats, see the [project README](README.md).

## What Git tracks

`.gitignore` excludes the local `imp/` environment, everything under `data/raw/`, and generated embedding files (`.npy`, `.npz`, model weights, and related artifacts). Only small, intentional split lists, metadata, reports, and result summaries are published. Empty placeholders for `data/raw/` and `artifacts/embeddings/` are not tracked.
