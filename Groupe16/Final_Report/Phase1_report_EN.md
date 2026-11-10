# A Comparative Study of RAG Techniques for Generating Image Captions

## Phase-1 Technical Report: Data Exploration & Pre-processing

**Master 1 – Intelligent Computer Systems Engineering (SII)**

**Module:** Introduction to Image Processing (ITI)

**University of Algiers 1 Benyoucef Benkhedda – Faculty of Sciences, Department of Computer Science**

**Project authors:** Hamou Yasmine, Ouziala Ali, Lounis Aya

**Academic year:** 2026–2027

**Milestone:** Updated Phase 1 Release

This Markdown edition follows the content and reported measurements of the submitted [PDF](Phase1_report_EN.pdf). Figures are linked from [`results/figures/`](../../results/figures/). The separately published retrieval-validation files are from a later run; see [`results/README.md`](../../results/README.md) for their timestamp and measurements.

## Contents

1. [Executive summary and pipeline](#1-executive-summary-and-pipeline)
2. [Step 1: Environment setup and data acquisition](#2-step-1-environment-setup-and-data-acquisition)
3. [Step 2: Dataset partitioning and anti-leakage](#3-step-2-dataset-partitioning-and-anti-leakage)
4. [Step 3: Data auditing and anomaly handling](#4-step-3-data-auditing-and-anomaly-handling)
5. [Steps 4–5: Image and caption pre-processing](#5-steps-4-5-image-and-caption-pre-processing)
6. [Step 6: Metadata and confidence policy](#6-step-6-metadata-and-confidence-policy)
7. [Step 7: Exploratory data analysis](#7-step-7-exploratory-data-analysis)
8. [Steps 8–9: Embeddings and FAISS indexing](#8-steps-8-9-embeddings-and-faiss-indexing)
9. [Step 10: Retrieval validation and qualitative analysis](#9-step-10-retrieval-validation-and-qualitative-analysis)
10. [Reproduction and Phase 2 roadmap](#10-reproduction-and-phase-2-roadmap)

## 1. Executive summary and pipeline

Automatic image caption generation requires fluent, semantically accurate descriptions. Vision-language models may hallucinate entities, omit small objects, or confuse spatial relationships. Retrieval-Augmented Generation (RAG) aims to ground caption generation in visually similar examples and human-validated reference captions.

This report documents Phase 1: data exploration and pre-processing. Its ten implemented steps are:

The report's scope is 10,000 knowledge-base images and 100 validation images from `train2017`, plus 1,000 test images from `val2017` with the original 100-image test retained as a comparison subset. It documents image-grounded annotations and the Step 5 confidence policy.

| Step | Pipeline phase | Core objective |
|---:|---|---|
| 1 | Data acquisition | Download and verify official MS COCO 2017 data and model resources. |
| 2 | Split partitioning | Create 10,000 knowledge-base and 100 validation images from `train2017`, plus 1,000 test images from `val2017`. |
| 3 | Data auditing | Check image decoding, SHA-256 integrity, duplicates, and anomalies. |
| 4 | Visual pre-processing | Apply aspect-preserving bicubic resizing, a central `224 × 224` crop, RGB conversion, and CLIP normalization. |
| 5 | Caption pre-processing | Normalize whitespace in a clean copy while preserving raw text. |
| 6 | Metadata extraction | Ground objects in bounding boxes and apply the documented human-reference confidence convention. |
| 7 | Statistical EDA | Analyze object distributions, scene complexity, vocabulary, and spatial relations. |
| 8 | Multimodal embeddings | Extract L2-normalized 512-dimensional CLIP ViT-B/32 vectors. |
| 9 | FAISS indexing | Build exact cosine-similarity indexes with `IndexFlatIP`. |
| 10 | Retrieval validation | Check retrieval on 100 disjoint validation queries and inspect qualitative cases. |

![Phase 1 pipeline](../../results/figures/phase1_pipeline_en.png)

*Figure 1. End-to-end Phase 1 execution architecture. Test reference captions remain isolated from retrieval.*

## 2. Step 1: Environment setup and data acquisition

The report records a Python 3.12 environment with pinned dependencies, including `torch 2.14.1+cpu`, `transformers 5.21.0`, `faiss-cpu 1.15.1`, `nltk 3.10.4`, and `pycocotools 2.0.11`.

Official Microsoft COCO S3 endpoints are used over HTTPS to obtain:

1. Caption and instance-annotation JSON files for `train2017` and `val2017`.
2. Images belonging to the frozen splits, stored under `data/raw/train2017/` and `data/raw/val2017/`.
3. The `openai/clip-vit-base-patch32` encoder, pinned in the report to revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
4. The English NLTK POS resource `averaged_perceptron_tagger_eng`, cached under `data/interim/cache/nltk_data/`.

## 3. Step 2: Dataset partitioning and anti-leakage

The splits are designed to prevent image-ID overlap between knowledge base, validation, and test:

| Partition | COCO source | Target images | Audited captions |
|---|---|---:|---:|
| Knowledge base | `train2017` | 10,000 | 50,000 |
| Validation subset | `train2017` | 100 | 500 |
| Preserved comparison test | `val2017` | 100 | 500 |
| Extended evaluation test | `val2017` | 1,000 | 5,000 |
| Total distinct `train2017` images | `train2017` | 10,100 | 50,500 |
| Total overall (comparison subset included in test) | `train2017` + `val2017` | 11,100 | 55,500 |

Let \(U_{\text{train}}\) and \(U_{\text{val}}\) be the eligible image pools:

- \(S_{\text{val}} \subset U_{\text{train}}\), with \(|S_{\text{val}}|=100\).
- \(S_{\text{KB}} \subset (U_{\text{train}}\setminus S_{\text{val}})\), with \(|S_{\text{KB}}|=10{,}000\), so \(S_{\text{KB}}\cap S_{\text{val}}=\varnothing\).
- The 1,000-image test set retains the original 100 and adds 900 stratified images from the remaining eligible `val2017` pool. The original 100 are recorded in `artifacts/splits/comparison/test_100_ids.json`.

Sampling uses seed 42 and largest-remainder proportional allocation across scene-complexity strata: Simple (0–3 objects), Medium (4–7), and Complex (8 or more).

## 4. Step 3: Data auditing and anomaly handling

The report records a data audit over 11,100 images and 55,500 captions:

| Audit check | Observed result | Status |
|---|---:|---|
| Expected image files | 11,100 | Target |
| Images present locally | 11,100 | 100% verified |
| Images opened and decoded by Pillow | 11,100 | Valid |
| Missing, unreadable, or truncated images | 0 | Passed |
| Captions per image | 5.00 average | Zero variance |
| Shared IDs between knowledge base and validation | 0 | Anti-leakage verified |
| Shared IDs between knowledge base and test | 0 | Anti-leakage verified |
| Duplicate file or pixel hashes | 0 | No duplication |
| Degenerate bounding boxes | 1 | Documented and handled |
| Identical normalized captions across different splits | 87 | Inspected; no image-ID leakage |

### 4.1 Documented anomalies

1. **Degenerate box in image 200365:** COCO annotation 918 has bounding box `[296.65, 388.33, 1.03, 0.0]`. The source annotation is not modified or deleted. The image, caption, and category are retained, and the box is marked `bbox_valid=false`; it is excluded from spatial geometry and IoU calculations.
2. **Repeated caption strings:** 87 normalized caption strings occur for different images across splits (for example, generic descriptions of people at a dining table or a train at night). The audit found no image-ID or annotation-ID leakage.

## 5. Steps 4-5: Image and caption pre-processing

### 5.1 Step 4: Image pre-processing

1. Apply EXIF orientation with `ImageOps.exif_transpose`, then convert to three-channel RGB.
2. Resize bicubically while preserving aspect ratio so the shorter dimension is 224 pixels.
3. Extract a centered `224 × 224` crop.
4. Scale intensities to `[0, 1]` and normalize using CLIP channel statistics:

   - Mean: `[0.48145466, 0.45782750, 0.40821073]`
   - Standard deviation: `[0.26862954, 0.26130258, 0.27577711]`

The resulting tensor has shape `(1, 3, 224, 224)` and finite floating-point values.

### 5.2 Step 5: Caption normalization

Raw reference captions are preserved for later evaluation (for example, BLEU, CIDEr, and SPICE). The clean version is prepared separately:

- `caption_raw`: original punctuation, capitalization, and spacing.
- `caption_clean`: repeated, leading, and trailing whitespace and line breaks normalized.

| Split | Total captions | Modified | Raw characters | Clean characters |
|---|---:|---:|---:|---:|
| Knowledge base | 50,000 | 7,743 | 2,620,933 | 2,612,839 |
| Validation | 500 | 72 | 25,725 | 25,645 |
| Test (1,000) | 5,000 | 745 | 262,492 | 261,674 |
| Test subset (100) | 500 | 88 | 25,680 | 25,584 |

## 6. Step 6: Metadata and confidence policy

This implementation calls metadata extraction Step 6 to reflect its ten-step pipeline; the instructor's Section 4.4.3 calls it Step 5. Extracted JSON Lines records in `artifacts/metadata/{split}.jsonl` associate image properties and SHA-256 fingerprints with five captions, visual objects, boxes, spatial relations, and heuristic attributes.

### 6.1 Human-reference confidence convention

- COCO human captions use `source="coco_human_reference"`, `confidence=1.0`, and `threshold=1.0`. This is a provenance convention, not a model-detection probability.
- Objects and boxes come from COCO instance annotations (`source="coco_annotation"`), rather than nouns extracted only from captions.
- Caption-derived attributes such as “white” or “blue” use `source="caption_heuristic"` and do not inherit the human-reference confidence value.

## 7. Step 7: Statistical exploratory data analysis

Statistics and figures are computed from the extracted metadata records.

### 7.1 Object classes and imbalance

The report counts 72,507 object annotations across the 80 COCO categories in the 10,000-image knowledge base:

| Rank | Category | Instances | Rank | Category | Instances | Rank | Category | Instances |
|---:|---|---:|---:|---|---:|---:|---|---:|
| 1 | person | 22,670 | 10 | handbag | 1,044 | 18 | sheep | 829 |
| 2 | car | 3,815 | 11 | bird | 950 | 19 | truck | 751 |
| 3 | chair | 3,151 | 12 | banana | 945 | 20 | potted plant | 752 |
| 4 | bottle | 2,086 | 13 | umbrella | 920 | 21 | motorcycle | 721 |
| 5 | book | 1,986 | 14 | bench | 869 | 22 | backpack | 706 |
| 6 | cup | 1,584 | 15 | boat | 829 | 23 | cow | 704 |
| 7 | dining table | 1,288 | 16 | wine glass | 673 | 24 | knife | 687 |
| 8 | bowl | 1,158 | 17 | donut | 662 | 25 | kite | 620 |
| 9 | traffic light | 1,112 | — | — | — | — | — | — |

“Person” represents 31.3% of all counted objects. Rare examples include toaster (17) and hair drier (20).

![Top object categories](../../results/figures/knowledge_base_top50_objects.png)

*Figure 2. Frequency distribution of the top 50 knowledge-base object categories.*

### 7.2 Scene complexity

Complexity is the number of annotated object instances per image.

| Metric | Knowledge base | Validation | Test (100) | Test (1,000) |
|---|---:|---:|---:|---:|
| Images | 10,000 | 100 | 100 | 1,000 |
| Mean objects/image | 7.2507 | 7.3400 | 7.0500 | 7.3880 |
| Median objects/image | 4.0 | 4.5 | 4.0 | 4.0 |
| Population standard deviation | 7.3305 | 8.3848 | 6.4628 | 7.4213 |
| Minimum–maximum | 0–59 | 0–58 | 1–26 | 0–63 |
| Simple scenes (0–3 objects) | 4,247 (42.47%) | 41 (41.00%) | 42 (42.00%) | 422 (42.20%) |
| Medium scenes (4–7 objects) | 2,449 (24.49%) | 25 (25.00%) | 24 (24.00%) | 238 (23.80%) |
| Complex scenes (≥8 objects) | 3,304 (33.04%) | 34 (34.00%) | 34 (34.00%) | 340 (34.00%) |

The report interprets the close Test-100 and Test-1000 distributions as evidence that proportional stratification preserved scene-complexity proportions.

![Test-100 and test-1000 comparison](../../results/figures/phase1_test100_test1000.png)

*Figure 3. Normalized statistical comparison between the original test-100 and extended test-1000 sets.*

### 7.3 Caption and POS statistics

Caption lengths and vocabulary were analyzed with NLTK's Treebank tokenizer:

| Linguistic property | Knowledge base | Validation | Test (100) | Test (1,000) |
|---|---:|---:|---:|---:|
| Reference captions | 50,000 | 500 | 500 | 5,000 |
| Mean words/caption | 10.4525 | 10.2900 | 10.2720 | 10.4490 |
| Median words/caption | 10.0 | 10.0 | 10.0 | 10.0 |
| Caption length range | 6–49 | 7–32 | 7–21 | 7–49 |
| Unique vocabulary tokens | 9,972 | 945 | 904 | 3,372 |
| Total word tokens | 522,625 | 5,145 | 5,136 | 52,245 |

Top knowledge-base Penn Treebank POS tags:

| Tag | Description | Count | % | Tag | Description | Count | % |
|---|---|---:|---:|---|---|---:|---:|
| NN | Singular noun | 149,690 | 28.6% | VBN | Past participle | 6,693 | 1.3% |
| DT | Determiner | 107,020 | 20.5% | CD | Cardinal number | 6,334 | 1.2% |
| IN | Preposition/conjunction | 83,586 | 16.0% | TO | Infinitival “to” | 5,743 | 1.1% |
| JJ | Adjective | 41,620 | 8.0% | VBP | Non-3rd-person verb | 5,524 | 1.1% |
| NNS | Plural noun | 30,721 | 5.9% | RP | Adverbial particle | 3,671 | 0.7% |
| VBG | Gerund/participle | 29,990 | 5.7% | PRP$ | Possessive pronoun | 3,646 | 0.7% |
| VBZ | 3rd-person singular verb | 13,688 | 2.6% | RB | Adverb | 3,525 | 0.7% |
| CC | Coordinating conjunction | 11,967 | 2.3% | PRP | Personal pronoun | 3,356 | 0.6% |
| NNP | Proper singular noun | 6,916 | 1.3% | WDT | Wh-determiner | 2,308 | 0.4% |

### 7.4 Co-occurrence, spatial relations, and derived attributes

- Across valid bounding-box pairs, the report records 495,256 ordered left/right pairs, 495,260 above/below pairs, and 45,020 box overlaps with IoU ≥ 0.1.
- Frequent caption color terms include white (1,997 images), black (1,123), red (870), and blue (780).
- Spatial terms include on (7,283 images), near (1,361), and behind (532). Context terms include outside (563), inside (400), and night (136).

![Object co-occurrence heatmap](../../results/figures/knowledge_base_cooccurrence.png)

*Figure 4. Co-occurrence heatmap for the top 20 object categories.*

## 8. Steps 8-9: Embeddings and FAISS indexing

### 8.1 CLIP vectors

The report describes `openai/clip-vit-base-patch32` embeddings generated on CPU in batches of 32:

- 10,000 image vectors with shape `(10000, 512)`.
- 50,000 caption vectors with shape `(50000, 512)`.
- Every vector is L2-normalized: \(v=e/\lVert e\rVert_2\), with \(\lVert v\rVert_2=1\).

### 8.2 FAISS indexes

`IndexFlatIP` performs exact inner-product search. Because query and stored vectors are L2-normalized, inner product equals cosine similarity:

\[
\langle q,v\rangle = \sum_{j=1}^{512}q_jv_j
=\frac{q\cdot v}{\lVert q\rVert_2\lVert v\rVert_2}
=\cos(\theta_{q,v}).
\]

| Index | Algorithm | Vectors | Dimensions | Build latency | Disk footprint |
|---|---|---:|---:|---:|---:|
| Image | `IndexFlatIP` | 10,000 | 512 | 0.0075 s | 20.48 MB |
| Caption | `IndexFlatIP` | 50,000 | 512 | 0.0340 s | 102.40 MB |

`image_rows.json` and `text_rows.json` map vector rows to COCO image IDs and reference captions.

## 9. Step 10: Retrieval validation and qualitative analysis

The report describes 100 disjoint validation images encoded as queries against the 10,000-image knowledge base, retrieving five neighbors each. Validation captions are withheld from search. It records a mean CPU search latency of 4.167 ms over 500 retrieved neighbors. This is a functional validation, not a final retrieval-relevance metric or caption-generation evaluation.

### 9.1 Case study: validation image 490

Image `000000000490.jpg` shows a dog with its front paws on a skateboard in a dim indoor room with a hardwood or parquet floor, a wall, and dark shadows.

| Rank | Retrieved COCO ID | Cosine similarity | Retrieved reference caption |
|---:|---:|---:|---|
| 1 | 267664 | 0.818381 | A lone cat sits in a dark room with some furniture. |
| 2 | 545323 | 0.801236 | A dog is sitting on a black bench. |
| 3 | 127643 | 0.790362 | A brown dog standing on top of a skateboard. |
| 4 | 516396 | 0.778416 | A dog standing with front paws on a skateboard. |
| 5 | 81686 | 0.772747 | A dog sits on a rug with its eyes closed. |

![Qualitative retrieval cases](../../results/figures/phase1_qualitative_cases.png)

*Figure 5. Qualitative retrieval results for validation query 490.*

Ranks 3 and 4 retrieve the compound “dog + skateboard” semantics, and rank 2 also identifies a dog. Rank 1 instead retrieves a cat in a dark room. The report attributes this failure to global CLIP features responding to scene-level appearance (lighting, shadows, floor reflections, and animal silhouette). It motivates Phase 2 methods such as reranking, multimodal query expansion, or metadata filtering.

The numbers in this section are transcribed from the submitted PDF. The separate current retrieval validation output is timestamped in [`results/retrieval/validation/results.json`](../../results/retrieval/validation/results.json); its measurements are intentionally not substituted into this report.

## 10. Reproduction and Phase 2 roadmap

From the repository root, the report's command sequence is:

```powershell
python -m venv imp
.\imp\Scripts\python.exe -m pip install -r requirements.txt
python -m src.cli download-annotations
python -m src.cli download-images
python -m src.cli download-nlp
python -m src.cli download-model
python -m src.cli audit
python -m src.cli metadata
python -m src.cli eda
python -m src.cli --offline embeddings
python -m src.cli index
python -m src.cli --offline validate-retrieval --top-k 5
python -m src.cli --offline qc
python -m src.cli report
```

The prepared split IDs should be reused; do not force-regenerate them. Phase 2 is planned to:

1. Integrate a generative VLM or LLM with retrieved multimodal context.
2. Implement the assigned RAG technique (for example, reranking, multimodal RAG, Self-RAG, or metadata-filtered RAG).
3. Evaluate generated captions against COCO references using BLEU-4, METEOR, ROUGE-L, CIDEr, and SPICE.
4. Apply statistical tests (for example, Shapiro–Wilk followed by a paired t-test or Wilcoxon signed-rank test, with Bonferroni correction).
