# Dataset Description

## 1. Dataset source

This project uses the **Microsoft Common Objects in Context (COCO)
2017** dataset for image-captioning and image-retrieval experiments.

COCO provides real-world images together with natural-language captions
and object annotations. The project uses images from the `train2017` and
`val2017` subsets.

-   **Training source:** `train2017`
-   **Test source:** `val2017`
-   **Image format:** JPEG
-   **Image content:** everyday scenes containing people, animals,
    objects, and indoor or outdoor environments.

The original COCO images and annotation files are not reproduced in this
repository. They must be obtained separately from the official COCO
dataset distribution.

## 2. Image subsets used in the project

The project defines the following principal subsets:

  ------------------------------------------------------------------------
  Subset           COCO source           Number of images Purpose
  ---------------- ---------------- --------------------- ----------------
  Knowledge Base   `train2017`                     10,000 Main image
                                                          collection used
                                                          to build the
                                                          retrieval
                                                          knowledge base

  Validation       `train2017`                        100 Validation and
                                                          technical checks

  Test             `val2017`                          100 Final test
                                                          subset
  ------------------------------------------------------------------------

The Knowledge Base and validation IDs were selected from `train2017`
after shuffling the image IDs with the fixed random seed `42`. The first
10,000 IDs were assigned to the Knowledge Base and the following 100 IDs
to validation.

The 100 test IDs were selected from `val2017`. The split checks
confirmed that the three principal subsets contain no overlapping image
IDs.

An additional file, `data/splits/test_1000_image_ids.json`, contains an
extended 1,000-image test selection. It is supplementary and should not
be confused with the 100-image test subset used by the principal split
configuration.

## 3. Captions

COCO provides five captions per image in the usual case. However, the
number of captions is not exactly five for every image.

For the 10,000 Knowledge Base images, the project contains:

-   **10,000 images**
-   **50,028 captions**
-   **9,972 images with five captions**
-   **28 images with six captions**
-   **No missing image references** in the caption preprocessing output

Two caption versions are retained:

-   **Raw captions:** the original caption text, preserved for reference
    and evaluation.
-   **Cleaned captions:** normalized text used by the text-processing
    and retrieval pipeline.

The cleaning stage includes lowercasing, trimming leading and trailing
whitespace, collapsing repeated whitespace, and removing spaces before
punctuation. The wording of the captions is otherwise retained.

## 4. Image preprocessing

The image preprocessing pipeline prepares the Knowledge Base images for
visual feature extraction.

Images are handled as RGB inputs. The CLIP processor performs the
model-specific resizing, cropping, normalization, and tensor conversion
required by the selected model. The original image aspect ratio is not
manually changed by an independent stretching operation in the CLIP
input pipeline.

The visual embedding model used in the project is **CLIP ViT-B/32**
(`openai/clip-vit-base-patch32`).

## 5. COCO object annotations and metadata

In addition to captions, COCO instance annotations are used to extract
structured information about objects appearing in the Knowledge Base
images.

The metadata extraction stage records information such as:

-   object categories;
-   bounding boxes;
-   object areas;
-   crowd indicators (`iscrowd`);
-   object counts per image;
-   spatial relationships estimated from object positions.

For the 10,000 Knowledge Base images, the extracted COCO metadata
contains **71,773 object instances** and approximately **961,525 spatial
relationships**. These values describe the metadata extraction output,
not the number of images or captions.

## 6. Additional YOLO detections

The project also includes an additional object-detection stage using
**YOLO26n**. This stage supplements the original COCO annotations; it
does not replace them.

The detector was run on the 10,000 Knowledge Base images with a
confidence threshold of `0.25`. The resulting outputs contain:

-   **49,375 detected objects**;
-   **79 categories represented among the detections**;
-   **136 images with no YOLO detection**.

The number 79 refers to categories actually present in the detector
output, not to the total number of COCO categories.

The merged metadata file `data/metadata/kb_metadata_with_yolo.json` is
approximately 143 MB and is not included in the Git repository because
of its size. The generated YOLO detection outputs are stored separately
under `data/visual_detections/full_10000/`, subject to the repository's
tracked-file configuration.

## 7. Image and text embeddings

For retrieval, CLIP is used to encode both images and captions into a
shared feature space.

  Embedding type       Number of vectors   Dimension Data type
  ------------------ ------------------- ----------- -----------
  Image embeddings                10,000         512 `float32`
  Text embeddings                 50,028         512 `float32`

These embeddings are used by the project's retrieval pipeline. The
512-dimensional representation is the output dimension of the selected
CLIP model.

## 8. Data organization

The main dataset-related files are organized under the `data/`
directory:

``` text
data/
├── splits/
│   ├── knowledge_base_image_ids.json
│   ├── validation_image_ids.json
│   ├── test_image_ids.json
│   └── test_1000_image_ids.json
├── captions/
│   ├── kb_captions_raw.json
│   └── kb_captions_cleaned.json
├── metadata/
│   └── kb_metadata.json
├── visual_detections/
│   └── full_10000/
└── embeddings/
```

The exact files present may vary as the pipeline evolves. Large source
images, generated intermediate files, and some large merged artifacts
may be excluded from Git tracking. The split ID files and preprocessing
outputs document which COCO images are used.

## 9. Reproducibility notes

-   The principal Knowledge Base and validation split uses random seed
    `42`.
-   The source subset for the Knowledge Base and validation is
    `train2017`.
-   The source subset for the principal test split is `val2017`.
-   The principal split sizes are 10,000 Knowledge Base images, 100
    validation images, and 100 test images.
-   COCO annotations and YOLO predictions are distinct sources of visual
    metadata and should be interpreted separately.
-   The repository stores project code, split definitions, and selected
    processing outputs; users may need to download the original COCO
    images and annotations separately.

## 10. Dataset reference

Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). **Microsoft COCO:
Common Objects in Context.** *European Conference on Computer Vision
(ECCV).*\
Project website: <https://cocodataset.org/>
