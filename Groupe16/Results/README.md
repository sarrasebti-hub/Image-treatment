# Results

This folder contains reviewable outputs copied from `reports/figures/` and the most recent `reports/retrieval/validation/` run. It excludes local image data, model weights, and `.npy` embedding arrays.

## Figures

[`figures/`](figures/) contains all 51 PNG EDA, split-comparison, pipeline, and qualitative figures produced by the project.

## Retrieval validation

[`retrieval/validation/`](retrieval/validation/) contains:

- `results.json` — run metadata, configuration and model revision, query summaries, and retrieved neighbors.
- `results.csv` — one row per query-neighbor pair, with rank and cosine similarity.
- `query_rows.json` — the 100 validation query image IDs.
- `gallery.html` — a local visual gallery. It refers to COCO JPEGs under `data/raw/train2017/`; those ignored images are not hosted in this repository, so the gallery requires the local dataset to be restored.

The derived `query_embeddings.npy` is intentionally not copied or committed; `.npy` files are excluded by `.gitignore`.

The included run, created `2026-11-10T15:47:33Z`, reports **passed**, 100 queries, a 10,000-image knowledge base, top-5 retrieval, and `openai/clip-vit-base-patch32` at revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Its recorded mean search latency is approximately 8.846 ms. This is functional validation only—not a final relevance score, caption-generation result, or Phase 2 evaluation.

The English report PDF contains an earlier run's measurements. See [the Markdown report](../output/pdf/Phase1_report_EN.md) for its source values and explanation.
