# Large artifacts kept on Google Drive (not in this repo)

Shared folder: https://drive.google.com/drive/folders/14v47jEKJ3rVawyNBtb_BAI6XFGH1MJT7  (`G9_Hybrid_RAG`)

Small files are already in this repo (same paths as on Drive): `data/splits/*`, `data/metadata/*.csv`, `embedding_config.json`, `results/qa/*` (except the image audit), `results/retrieval/*.csv|json`, `results/preprocessing_release/*`.

| Drive path | Contents | Why not in git |
|---|---|---|
| `data/annotations/` | official COCO JSON files | large, downloadable from COCO |
| `data/approved_annotations/` | `kb_captions.json`, `kb_instances.json`, `validation_*.json` | ~MBs; rebuilt by notebook 01 |
| `data/preprocessed_captions/` | `kb_captions_retrieval.csv` (SHA-256 in `results/preprocessing_release/small_file_sha256.json`) | copy into repo if wanted (~MBs) |
| `data/images_archives/` | uncompressed ZIPs of 10,200 images | too large |
| `data/embeddings/clip_vit_base_patch32/` | `kb_image_embeddings.npy` (10,000×512), `kb_text_embeddings.npy` (50,000×512), `kb_image_ids.npy`, `kb_text_metadata.csv`, shard folders | large |
| `data/indexes/` | `faiss_images.index` (20 MB), `faiss_captions.index` (102 MB), `bm25.pkl` (4.7 MB) | large |
| `data/metadata/image_metadata.json` | full per-image records (6.6 MB) | too big for the Drive connector used for this sync; the flat `image_metadata.csv` is in the repo. Copy by hand if wanted. |
| `results/eda/` | full EDA outputs incl. PNG charts and big CSVs | core stats copied into `results/eda/` |
| `results/qa/image_preprocessing_audit.csv` | 10,200-row image audit (0.9 MB) | copy if needed |
| `results/retrieval/image_query_hybrid_visualization.png` | retrieval demo figure (9 MB) | large |

⚠️ `data/embeddings/clip_vit_base_patch32/embedding_config.json` in **this repo is the corrected one** (generation version 5.20.2). Copy it to Drive before re-running notebook 06.
