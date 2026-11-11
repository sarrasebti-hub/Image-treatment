# Embeddings ready for retrieval (Phase 1)

Release `phase1-embeddings-v1`, built from the frozen splits in `cleaned_dataset/`
(10,000 knowledge-base images with 50,036 captions, 100 validation images inside the
knowledge base, 100 separate test images). It provides CLIP image and caption vectors,
Sentence-BERT caption vectors, exact FAISS indexes, row-to-ID maps and a small Python API
(`retrieval.py`) that applies the anti-leakage rules.

## Quick start

```bash
pip install -r requirements.txt
python build_embeddings.py     # first run downloads ~700 MB of pinned model weights
python check_embeddings.py     # 20 hard checks, leakage audit, validation metrics, figures
```

The build needs the split images in `cleaned_dataset/*/images/` (git-ignored, shared
separately). It refuses to overwrite an existing release unless `--overwrite` is passed.
On the reference machine (RTX 3060) the build took about 1.5 minutes after the download.

```python
from retrieval import KnowledgeBase   # from notebooks/: import sys; sys.path.insert(0, "..")

kb = KnowledgeBase()                   # loads indexes and ID maps; encoders load on first use

# Test image -> similar KB images, their captions, and KB captions directly (CLIP space)
query = kb.encode_images([kb.image_path(3156)])
images = kb.search_images(query, k=5)[0]                     # [Hit(image_id, score), ...]
evidence = [kb.captions_for_image(h.image_id) for h in images]
captions = kb.search_captions(query, k=5, space="clip")[0]   # Hit(image_id, score, caption_id, caption)

# Validation image (it is inside the KB): always exclude the query image and its captions
val_id = int(kb.validation_image_ids[0])
query = kb.image_embeddings("validation")[:1]
images = kb.search_images(query, k=5, exclude=[{val_id}])[0]

# Text query -> KB captions (Sentence-BERT space, e.g. a draft caption or an expanded query)
hits = kb.search_captions(kb.encode_texts(["a man riding a wave on a surfboard"], space="sbert"),
                          k=5, space="sbert")[0]
```

Always embed queries through `kb.encode_images` / `kb.encode_texts`. They apply exactly
the preprocessing used for the knowledge base. `check_embeddings.py` verifies this: re-encoding
the validation and test images and 200 captions reproduces the stored vectors with cosine
≥ 0.9999997. Queries encoded on CPU
(`KnowledgeBase(device="cpu")`) also match the GPU-built vectors. On 20 validation images and
50 captions the minimum cosine was 0.9999999, and the self-excluded top-5 lists were identical,
so no GPU is needed to use the index.

Two rules for callers:

- `exclude` takes one collection of integer image IDs per query: `exclude=[{image_id}]`. A bare
  string or int raises `TypeError`, because `exclude=["23686"]` would otherwise exclude the
  wrong IDs.
- Import `retrieval` before any `import faiss` of your own. On Windows, the order torch → faiss →
  transformers model code crashes Python with a segfault. `retrieval` imports transformers before
  faiss to avoid this. These orders were tested and work: `retrieval` first; `import torch` first;
  faiss imported before torch (as the two scripts do); and loading other transformers models
  (BLIP-2, T5) after `retrieval`.

## Files

| Path | Content | In git |
|---|---|---|
| `clip_vit_b32/kb_image_embeddings.npy` | 10,000 × 512 CLIP image vectors, row i = `kb_image_ids.json[i]` | no (19.5 MiB) |
| `clip_vit_b32/test_image_embeddings.npy` | 100 × 512 CLIP image vectors, row i = `test_image_ids.json[i]`; queries only | no (0.2 MiB) |
| `clip_vit_b32/kb_caption_embeddings.npy` | 50,036 × 512 CLIP text vectors, row i = `kb_captions.csv` row i | no (97.7 MiB) |
| `minilm_l6_v2/kb_caption_embeddings.npy` | 50,036 × 384 Sentence-BERT vectors, same rows | no (73.3 MiB) |
| `clip_vit_b32/kb_image.faiss`, `clip_vit_b32/kb_caption.faiss`, `minilm_l6_v2/kb_caption.faiss` | FAISS indexes over the three KB matrices | no (190.6 MiB) |
| `kb_image_ids.json`, `test_image_ids.json` | COCO image ID of every image row | yes |
| `validation_kb_rows.json` | the 100 validation image IDs and their KB row (validation vectors are KB rows, not a copy) | yes |
| `kb_captions.csv` | `row, image_id, caption_id, caption_raw, caption_embedded` for every caption row | yes |
| `metadata.json` | models, pinned revisions, settings, timings, environment, SHA256 of every binary | yes |
| `qc_report.json` | all check results, leakage audit, validation metrics, latency | yes |
| `figures/retrieval_examples.png`, `figures/leakage_top_pairs.png` | qualitative retrieval and leakage-review figures | yes |

Every vector links to a COCO image ID. Captions come from `kb_captions.csv` or
`kb.captions_for_image(image_id)`. Object annotations (categories, boxes) for the same image IDs
are in `cleaned_dataset/<split>/instances.json`.

The `.npy` and `.faiss` files are git-ignored by the repository's `*.npy` / `*.faiss` rules.
Share them like the images, for example as a GitHub Release asset, or rebuild them. To use
shared copies, place them in this folder and run `python check_embeddings.py`. It verifies each
file against the SHA256 values in the committed `metadata.json`. On the reference machine, a
from-scratch rebuild reproduced all 7 binaries and the 4 map files bit for bit (identical SHA256).
A rebuild on other hardware can differ at float precision, so it writes new checksums.

## Configuration

| Item | Value |
|---|---|
| Image encoder | `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Text encoders | CLIP text tower (same model, shared image-text space); `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Embedding dimensions | CLIP 512, MiniLM 384 |
| Image preprocessing | EXIF orientation (`ImageOps.exif_transpose`; 0 of 10,100 images needed it), RGB, CLIP processor: shortest side resized to 224 (bicubic, aspect ratio kept), 224 × 224 center crop, CLIP mean/std normalisation |
| Caption text embedded | original caption with whitespace collapsed (7,461 captions changed); case and punctuation kept. No caption truncated: longest is 55 CLIP / 56 MiniLM tokens (limits 77 / 256) |
| MiniLM pooling | attention-masked mean pooling, as in the model's sentence-transformers config, then L2 |
| Normalisation | every vector L2-normalised (unit norm), float32 |
| Batch sizes | 128 images, 512 texts; fp32 on GPU, TF32 off, cuDNN deterministic |
| Index type | `faiss.IndexFlatIP` (exact search; IVF/HNSW are unnecessary at 10k/50k vectors) |
| Similarity metric | inner product on unit vectors = cosine similarity |
| Database size | 10,000 image vectors + 2 × 50,036 caption vectors; 190.6 MiB of indexes |
| Indexing time | 0.010 s (images), 0.043 s (CLIP captions), 0.027 s (MiniLM captions) |
| Encoding time (RTX 3060) | 46.9 s KB images, 0.4 s test images, 28.3 s CLIP captions, 9.7 s MiniLM captions |

Model weights are fetched by pinned commit with `huggingface_hub.snapshot_download` and loaded
from the local folder. Their SHA256 is stored in `metadata.json`.

## Anti-leakage rules and audit

- Test captions are evaluation references only. `build_embeddings.py` never reads them, so no
  test caption is embedded or indexed. Test images are embedded as queries only and are not in
  any index, so a test image can never be retrieved for itself.
- Validation images are KB images. Retrieval for a validation query must pass
  `exclude=[{image_id}]`. That removes the image and all its captions from the candidates.
- Hard checks (all pass): test ∩ KB = ∅; validation ⊂ KB; no test caption ID or test image ID
  in `kb_captions.csv`; with exclusion, 100 validation queries never get their own image or
  captions back at k = 10 in any of the three indexes.
- Identical wording: 9 of the 500 test captions also occur verbatim among KB captions written for
  other images (generic phrasings such as "A man riding on the back of a motorcycle."). The KB is
  left unchanged, because filtering it with test references would itself use them.
- Near-duplicate scan (the exact-hash cleaning step could not do this): CLIP image cosine of every
  test and validation image against all 10,000 KB images. Test→KB nearest-neighbour cosine has
  median 0.846 and maximum 0.980. Two pairs reached the 0.95 review threshold and were inspected
  at full size:
  - test 183127 ↔ KB 379488 (0.980): same surf spot and framing, but a different surfer and board.
    These are two photos from one session, not the same image.
  - test 50943 ↔ KB 103260 (0.953): two different big-wave surfing photos.

  Validation→other-KB maximum is 0.946, with no pair flagged. The splits were not changed. For a
  stricter protocol the team can pass `exclude=[{379488}]` for test image 183127.
  See `figures/leakage_top_pairs.png`.

## Validation retrieval sanity check

100 validation queries, scored against the full KB (10,000 images / 50,036 captions). This
measures whether the embeddings and ID links work. It does not measure RAG quality. The test split
is not used for any retrieval metric.

| Task | Relevant items | R@1 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|
| Image → captions (CLIP) | the image's own captions (not excluded here) | 0.37 | 0.63 | 0.75 | 0.51 |
| Caption → image (CLIP) | the caption's image | 0.32 | 0.44 | 0.50 | 0.39 |
| Caption → captions (MiniLM) | the other captions of the same image | 0.19 | 0.43 | 0.55 | 0.30 |

With the query excluded, image→image neighbours share COCO object categories far above chance.
The mean category-set Jaccard is 0.50 at k = 1, 0.47 at k = 5 and 0.46 at k = 10, against 0.066 for
a random KB image (99 queries with at least one object). With 100 queries, a recall of 0.37 has a
standard error of about ±0.05.

Median single-query latency (RTX 3060 encoders, FAISS on 16 CPU threads): image-index search
0.9 ms, CLIP caption-index search 5.0 ms, MiniLM caption-index search 3.5 ms (k = 10); encoding
one image 13.2 ms, one CLIP text 6.6 ms, one MiniLM text 3.4 ms. These vary by a few percent
between runs; `qc_report.json` holds the values from the last check.

## Limits

- Center crop removes the edges of non-square images. This is the encoder's standard preprocessing.
- CLIP ViT-B/32 is the Phase 1 default. The encoder ablation (CLIP vs SigLIP vs DINOv2) needs a new
  `EncoderSpec` in `retrieval.py` and its own output folder.
- The near-duplicate threshold (0.95) is a heuristic for manual review, not a proof that no
  near-duplicates exist below it.
- The category Jaccard uses ground-truth COCO annotations only as a relevance proxy for this
  sanity check.
