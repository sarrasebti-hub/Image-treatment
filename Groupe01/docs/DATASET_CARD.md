# G1 shared dataset card — release `g1-v1.1`

## Release identity

The frozen manifest identifies release `g1-v1.1`, primary encoder `siglip-b16`,
seed `42`, and split sizes of 10,000 KB, 100 validation, and 100 test images
([`artifacts/manifest.json`](../artifacts/manifest.json)). The split metadata
records the same counts and ID hashes ([`artifacts/splits/splits_meta.json`](../artifacts/splits/splits_meta.json)).
The separate confirmation record is
[`reports/split_confirmation.json`](../reports/split_confirmation.json).

Use the frozen split files:

| Purpose | File |
|---|---|
| Knowledge base IDs and stratum fields | [`artifacts/splits/kb.json`](../artifacts/splits/kb.json) |
| Validation IDs and stratum fields | [`artifacts/splits/val.json`](../artifacts/splits/val.json) |
| Test IDs and stratum fields | [`artifacts/splits/test.json`](../artifacts/splits/test.json) |
| Sizes, seed, ID hashes, and observed strata | [`artifacts/splits/splits_meta.json`](../artifacts/splits/splits_meta.json) |
| Normalized RGB image files | `artifacts/images_rgb/<kb|val|test>/<12-digit-image-id>.jpg` |

The split files contain image IDs, source COCO split, filename, dimensions, and
the complexity/context/interaction sampling labels. Validation and test
references are separate from KB captions; do not merge them into a retrieval
index.

## Table schemas and real examples

### `images.csv`

One row per selected image. The CSV header fields, in order, are:

`image_id`, `split`, `source`, `file_name`, `status`, `note`, `width_orig`,
`height_orig`, `width`, `height`, `aspect`, `mode_orig`, `exif_orientation`,
`exif_corrected`, `converted_to_rgb`, `resized`, `sha256_raw`, `dhash`,
`brightness`, `contrast`, `colorfulness`, `sharpness`, `is_dark`,
`dominant_colors`, `color_hist`, `path_rgb`, `near_dup_flag`.

Real example (the first data row; values are kept as CSV strings):

```text
image_id=78, split=kb, source=train2017, file_name=000000000078.jpg,
status=ok, width_orig=612, height_orig=612, width=612, height=612,
aspect=1.0, mode_orig=RGB, exif_orientation=1, exif_corrected=False,
converted_to_rgb=False, resized=True,
sha256_raw=fbf237b294405624b5d662fe7f45a4c1b77bc275ade45f0737f5a6f6b1b6bb19,
dhash=1111000111100001110000011100001111000011110010011100100101001001,
brightness=0.593, contrast=0.1761, colorfulness=43.46, sharpness=0.015357,
is_dark=False, dominant_colors=gray|orange|red,
path_rgb=images_rgb\kb\000000000078.jpg, near_dup_flag=False
```

The source row is in [`artifacts/tables/images.csv`](../artifacts/tables/images.csv).
The table also includes the original color histogram and note fields; they are
omitted above only to keep the example readable.

### `kb_captions.jsonl`

Fields: `caption_id`, `image_id`, `caption_raw`, `caption_clean`, `n_tokens`.
There are five selected KB captions per KB image. Real row from
[`artifacts/tables/kb_captions.jsonl`](../artifacts/tables/kb_captions.jsonl):

```json
{"caption_id":755795,"image_id":78,"caption_raw":"A lighted owl candle sits next to a clock.","caption_clean":"a lighted owl candle sits next to a clock","n_tokens":9}
```

### `eval_refs_val.jsonl` and `eval_refs_test.jsonl`

Fields: `caption_id`, `image_id`, `caption_raw`. These files contain the
evaluation references only; each split has five references per image.

Real first rows from
[`artifacts/tables/eval_refs_val.jsonl`](../artifacts/tables/eval_refs_val.jsonl)
and
[`artifacts/tables/eval_refs_test.jsonl`](../artifacts/tables/eval_refs_test.jsonl):

```json
{"caption_id":810204,"image_id":650,"caption_raw":"a cat is sitting on top of a vehicle"}
{"caption_id":647108,"image_id":2261,"caption_raw":"A man on a blue raft attempting to catch a ride on a large wave."}
```

### `metadata.jsonl`

Top-level fields: `image_id`, `split`, `n_objects`, `category_counts`,
`n_categories`, `supercategories`, `complexity`, `context`, `day_night`,
`brightness`, `colorfulness`, `dominant_colors`, `objects`, `relations`,
`caption_attributes`, `metadata_sources`.

Real first row from
[`artifacts/tables/metadata.jsonl`](../artifacts/tables/metadata.jsonl):

```json
{"image_id":78,"split":"kb","n_objects":1,"category_counts":{"clock":1},"n_categories":1,"supercategories":["indoor"],"complexity":"simple","context":"indoor","day_night":"day_or_bright","brightness":0.593,"colorfulness":43.46,"dominant_colors":["gray","orange","red"],"objects":[{"category_id":85,"category":"clock","supercategory":"indoor","bbox_norm":[0.5879,0.0034,0.9378,0.3889],"area_frac":0.13487,"size":"medium","source":"coco_gt","dominant_color":"gray"}],"relations":[],"caption_attributes":[{"attr":"white","type":"color","head":"clock"}],"metadata_sources":{"objects":"coco_gt_annotations","attributes_pixel":"named_colors_hsv","relations":"bbox_geometry"}}
```

Nested object records include COCO category ID/name/supercategory, normalized
box, area fraction, size, source, and (where available) a dominant pixel color.
Relations are geometric labels. Caption-derived attributes are `null` for
validation/test and are present only for KB images.

## Embeddings

All saved encoder image/text embeddings are `float32`, L2-normalized, and row
aligned with their corresponding ID arrays. Shapes and dtypes below were read
from the frozen `.npy` files and checked against the normalization stated in
[`artifacts/index/index_report.json`](../artifacts/index/index_report.json).

| Encoder | File | Shape | Dtype | Normalization |
|---|---|---:|---|---|
| CLIP ViT-B/32 | `artifacts/embeddings/clip-b32/kb_img.npy` | `(10000, 512)` | float32 | L2 |
| CLIP ViT-B/32 | `artifacts/embeddings/clip-b32/val_img.npy` | `(100, 512)` | float32 | L2 |
| CLIP ViT-B/32 | `artifacts/embeddings/clip-b32/test_img.npy` | `(100, 512)` | float32 | L2 |
| CLIP ViT-B/32 | `artifacts/embeddings/clip-b32/kb_txt.npy` | `(50000, 512)` | float32 | L2 |
| DINOv2-small | `artifacts/embeddings/dinov2-s/kb_img.npy` | `(10000, 384)` | float32 | L2 |
| DINOv2-small | `artifacts/embeddings/dinov2-s/val_img.npy` | `(100, 384)` | float32 | L2 |
| DINOv2-small | `artifacts/embeddings/dinov2-s/test_img.npy` | `(100, 384)` | float32 | L2 |
| SigLIP base patch16/224 | `artifacts/embeddings/siglip-b16/kb_img.npy` | `(10000, 768)` | float32 | L2 |
| SigLIP base patch16/224 | `artifacts/embeddings/siglip-b16/val_img.npy` | `(100, 768)` | float32 | L2 |
| SigLIP base patch16/224 | `artifacts/embeddings/siglip-b16/test_img.npy` | `(100, 768)` | float32 | L2 |
| SigLIP base patch16/224 | `artifacts/embeddings/siglip-b16/kb_txt.npy` | `(50000, 768)` | float32 | L2 |
| SBERT all-MiniLM-L6-v2 | `artifacts/embeddings/sbert_kb_txt.npy` | `(50000, 384)` | float32 | L2 |

Each image encoder directory also has `ids_kb.npy`, `ids_val.npy`, and
`ids_test.npy` (`int64`, respectively 10,000, 100, and 100 IDs). CLIP and
SigLIP text arrays also have `kb_txt_caption_ids.npy` and
`kb_txt_image_ids.npy`; these each have 50,000 aligned `int64` rows.

## FAISS indexes

Image index characteristics and build durations below are from
[`artifacts/index/index_report.json`](../artifacts/index/index_report.json).
The report labels the exact index `IndexFlatIP`; `04_embeddings_index.ipynb`
constructs the approximate counterpart as `IndexHNSWFlat`. Both use inner
product over L2-normalized vectors (cosine similarity).

| Encoder | File suffix | Type | Dimension | Metric | Size (MB, report) | Build time (s, report) |
|---|---|---|---:|---|---:|---:|
| CLIP | `_flatip.faiss` | IndexFlatIP | 512 | inner product = cosine | 20.48 | 0.0081 |
| CLIP | `_hnsw.faiss` | IndexHNSWFlat | 512 | inner product = cosine | 23.198 | 4.006 |
| DINOv2 | `_flatip.faiss` | IndexFlatIP | 384 | inner product = cosine | 15.36 | 0.0497 |
| DINOv2 | `_hnsw.faiss` | IndexHNSWFlat | 384 | inner product = cosine | 18.078 | 6.8016 |
| SigLIP | `_flatip.faiss` | IndexFlatIP | 768 | inner product = cosine | 30.72 | 0.0225 |
| SigLIP | `_hnsw.faiss` | IndexHNSWFlat | 768 | inner product = cosine | 33.438 | 1.439 |

CLIP and SigLIP also have `_txt_flatip.faiss` indexes. The notebook constructs
these as `IndexFlatIP`; their dimensions are 512 and 768, respectively, and
their text embeddings are L2-normalized. The report records their text vector
shapes and text-embedding time, but does **not** record a separate text-index
build time or size entry. The following file sizes were read from the frozen
index files themselves, not inferred from the image-index report:

| Encoder | File | Type | Dimension | Metric | File size | Text-index build time |
|---|---|---|---:|---|---:|---|
| CLIP | `artifacts/index/clip-b32_txt_flatip.faiss` | IndexFlatIP | 512 | inner product = cosine | 102,400,045 bytes | not recorded |
| SigLIP | `artifacts/index/siglip-b16_txt_flatip.faiss` | IndexFlatIP | 768 | inner product = cosine | 153,600,045 bytes | not recorded |

Those sizes are filesystem byte counts; `index_report.json` does not provide
text-index size or type fields, and the text-index build time must not be
inferred from the text-embedding duration or image-index build fields. DINOv2
has no text tower index; SBERT is supplied as an embedding array, not a FAISS
index in this release.

## Load the release with `rag_data.Release`

Run this five-line example from the repository root; it loads the configured
primary encoder and performs an anti-self-retrieval query:

```python
from rag_data import Release
r = Release()
q = r.image_embeddings("val")[:1]
scores, ids = r.search_images(q, k=1, query_ids=r.ids("val")[:1])
print(r.encoder, scores.shape, ids.shape)
```

Observed output:

```text
siglip-b16 (1, 1) (1, 1)
```

To select another available image encoder explicitly, pass for example
`Release(encoder="clip-b32")`.

## Anti-leakage and reproducibility

- The saved split IDs are pairwise disjoint; validation and test images are not
  in the KB.
- Only `kb_captions.jsonl` and KB embeddings/indexes are retrieval data.
  Evaluation references are separate files and are used only for scoring.
- `Release.search_images(..., query_ids=...)` rejects a query image returned as
  its own KB result. Caption IDs are also checked separately by QC.
- Exact file duplicates are checked by SHA-256. Near-duplicate screening uses
  primary-encoder cosine similarity and perceptual dHash; details and current
  results are in [`reports/leakage_audit.md`](../reports/leakage_audit.md).
- The split seed is 42 (see the manifest and `splits_meta.json`).
- Verify the frozen release from the repository root with:

  ```powershell
  .\.venv\Scripts\python.exe tools\run_freeze_only.py --verify
  ```

  The verifier checks the SHA-256 values in `artifacts/manifest.json`; that
  manifest lists 10,270 hashed files.

## Known limitations

- Metadata object labels for validation/test come from COCO ground-truth
  instance annotations (an oracle for a deployed query); use the optional
  DETR object-detection path when evaluating realistic predicted query metadata.
- Caption-string matches across splits can occur for generic descriptions of
  different images; QC reports these separately from image/ID leakage.
- The recorded near-duplicate cosine check uses the configured primary
  encoder, SigLIP, rather than every encoder independently.
- `metadata.jsonl` spatial relations are computed from bounding-box geometry,
  not a semantic scene graph. Caption-derived attributes use a fixed lexical
  list and are KB-only.
- Only the first five captions, sorted by COCO caption ID, are retained where
  an image has more captions.
- The manifest records Python 3.13.7 for g1-v1.1. A separate provenance note
  provided for this documentation says Python 3.13.3; that discrepancy is
  unresolved. v1.0 runtime values are preserved in
  `artifacts/manifest_v1.0_backup.json`.
- Frozen artifact files are read-only after release freeze. Treat any
  re-embedding or data change as a new release; do not run artifact-producing
  pipeline stages against this release in place.



