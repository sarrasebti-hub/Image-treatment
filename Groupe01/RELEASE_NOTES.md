# G1 data release g1-v1.2

Prebuilt data for the shared preprocessing of the RAG image-captioning study (Group G1).
Code, splits, metadata, captions, EDA and reports are in the repository; this release holds the
large binary files that are not versioned in Git.

## Test sets

- **Official test: 1,000 images** (COCO val2017), `experiments/test_1000_val2017/`.
- Legacy test: 100 images (release `g1-v1.1`), kept unchanged.
- Knowledge base: 10,000 COCO train2017 images (50,000 captions); validation: 100 images.

## Assets

| Asset | Content | Restore to |
|---|---|---|
| `g1-v1.2-embeddings.zip` | CLIP-B/32 (512-d), SigLIP-B/16 (768-d), DINOv2-S (384-d) image embeddings for KB / val / legacy test, text embeddings, SBERT embeddings | repository root |
| `g1-v1.2-faiss-indexes.zip` | FAISS flat inner-product (exact, cosine on L2-normalised vectors) and HNSW (M=32, efC=200, efS=64) indexes | repository root |
| `g1-v1.2-test1000-embeddings.zip` | Image embeddings of the official 1,000-image test, 3 encoders | repository root |
| `g1-v1.2-images-rgb.zip` (optional) | Normalised RGB images (max side 512, JPEG q95) | repository root |
| `SHA256SUMS.txt` | SHA-256 of each asset | — |

## Use

    unzip g1-v1.2-embeddings.zip -d .          # archives contain repo-relative paths
    unzip g1-v1.2-faiss-indexes.zip -d .
    unzip g1-v1.2-test1000-embeddings.zip -d .
    sha256sum -c SHA256SUMS.txt                # Windows: certutil -hashfile <file> SHA256
    python scripts/qc_test_1000.py             # the SKIP checks now run

Embeddings are L2-normalised float32. Notes on the encoder choice and on the corrected official-test figures are in `README.md` and `PHASE1_REPORT.md` (section 11.3); the PDF report was recompiled after the correction and shows the corrected figures.

## Known deviations (reproducibility)

- Embeddings and indexes restored from the original working copy (rag_preproc) were compared with the frozen manifest (read-only check, verify_manifest.py). CLIP, DINOv2 and SBERT embeddings and the CLIP/DINOv2 indexes match (22 embedding files ok, 6 index files ok). The 4 SigLIP embedding files (kb_img, kb_txt, val_img, test_img) and the 3 SigLIP indexes (flat, hnsw, txt_flat) do NOT match the manifest; the cause is unknown (SIGLIP_ENV_NOTE.txt mentions Python 3.13.3 versus 3.13.7 in the manifest, which is not proven to be the cause). On the first 20 rows, SigLIP kb_txt matches a recomputation with padding="max_length" (64 tokens), as in notebook 04. The manifest and index_report.json were not modified; bit-for-bit SigLIP reproducibility is therefore not guaranteed.
- Official 1,000-image test (scripts/evaluate_test_1000.py): the committed results (experiments/test_1000_val2017/results/encoder_comparison_test_1000.csv/.json/.png) were REPLACED in this release by a re-evaluation that uses SigLIP text embeddings computed with padding="max_length" (64 tokens), as in notebook 04. SigLIP img->txt: CIDEr 0.6385, Recall@1 0.973, MRR 0.9847, the highest of the five and consistent with validation (0.7223) and the legacy 100-image test (0.7064). The earlier committed figures (CIDEr 0.4645, Recall@1 0.891, MRR 0.9376, lowest of the five; CLIP img->txt highest at 0.5746) are reproduced exactly by SigLIP kb_txt embeddings computed with padding=True, which is what scripts/recover_kb_embeddings.py does (measured: mean cosine similarity between the two kb_txt sets 0.4842, 0 of 50,000 rows above 0.9999). Which of the two sets matches the SHA256 in the manifest is NOT established; the max_length set follows notebook 04. Other configurations: CLIP img->txt (0.5746) and SigLIP img->img (0.5320) are identical; CLIP img->img (0.4978 -> 0.4984) and DINOv2 img->img (0.5333 -> 0.5309) changed slightly, cause not established. The latency columns were re-measured and differ (machine-dependent). The encoder selected on validation (siglip-b16) was not changed. The PDF report (reports/phase1_technical_report.pdf, section 11.3) was recompiled after this correction and shows the corrected figures (SigLIP img->txt 0.6385, highest of the five).
- 11 of the 10,200 normalised RGB images (all in the knowledge base, narrow images 207-316 x 640 px) differ from the manifest although the raw files are identical (sha256_raw). The cause is unknown.
- artifacts/eda was restored to the g1-v1.1 state (commit 15cb295 had regenerated it afterwards); its version label therefore reads g1-v1.0 in EDA_REPORT.md.
- The official 1,000-image test was confirmed orally by the instructor only (no written confirmation).
- Windows PowerShell: use Expand-Archive and Get-FileHash -Algorithm SHA256 instead of unzip and sha256sum.
