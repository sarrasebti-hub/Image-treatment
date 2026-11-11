# Member 4 — Exploration, Analysis, and Repository Documentation

## 1. Caption Dataset Exploration

The knowledge base initially contained 10,000 images and 50,000 captions. A leakage audit identified 223 captions that were excluded to reduce possible overlap-related contamination between dataset partitions.

After exclusion, the knowledge base contained 49,777 captions associated with the same 10,000 images. The cleaned dataset was saved as `data/captions_processed_v1.1.json`.

### 1.1 Caption Statistics Before and After Exclusion

| Statistic | Before exclusion | After exclusion |
|---|---:|---:|
| Number of images | 10,000 | 10,000 |
| Number of captions | 50,000 | 49,777 |
| Average words per caption | 10.4643 | 10.4694 |
| Unique captions | 49,372 | 49,268 |

The exclusion removed 223 captions. The average caption length changed only slightly, from approximately 10.46 to 10.47 words. This indicates that the exclusion primarily affected the number of captions rather than substantially changing their average length.

This comparison measures the effect of leakage exclusion. It should not be interpreted as a comparison of raw text against linguistically normalized or cleaned text.

The corresponding results are saved in:
- `exploration/caption_statistics.csv`
- `exploration/caption_exclusion_comparison.csv`


### 1.2 Caption Preprocessing and Vocabulary Analysis

The v1.1 exploration summary reports the following caption preprocessing statistics:

| Statistic | Before preprocessing | After preprocessing |
|---|---:|---:|
| Number of captions | 50,000 | 49,777 |
| Vocabulary size | 15,992 | 9,799 |
| Average words per caption | 10.464 | 10.478 |

The reported vocabulary decreased from 15,992 to 9,799 unique words. This suggests reduced vocabulary variation in the processed captions. The caption counts match the exclusion totals reported in Section 1.1, so they should not be interpreted as an independent measurement of caption removal during text preprocessing.

The results are recorded in `exploration/exploration_summary.json`.

### 1.3 Relationship Between Scene Complexity and Caption Length

A scatter plot was generated to examine the relationship between the number of detected objects and the average caption length.

The Pearson correlation coefficient was **0.0934**, indicating a weak positive relationship. In this analysis, images containing more detected objects do not necessarily have substantially longer captions.

The results are saved in:
- `exploration/06_scatter_correlation.csv`
- `exploration/visualizations/fig09_objects_vs_caption_length.png`


### 1.4 Part-of-Speech Distribution

A Part-of-Speech (POS) analysis was performed on a sample of 10,000 cleaned captions. The words were categorized into grammatical groups, including nouns, verbs, adjectives, adverbs, determiners, prepositions, conjunctions, numerals, and pronouns.

The analysis provides insight into the linguistic structure of the image captions. Nouns and verbs are particularly relevant because they describe objects, people, and actions in images.

The results are saved in:
- `exploration/07_pos_distribution.csv`
- `exploration/visualizations/fig10_pos_distribution.png`

## 2. Object Detection Metadata Analysis

Object-level metadata was generated using YOLOv8m with a confidence threshold of 0.25. The metadata file contains records for the knowledge base, validation, and test images.

### 2.1 Detection Statistics

| Metric | Result |
|---|---:|
| Metadata image records | 11,100 |
| Total detected objects | 78,381 |
| Number of detected classes | 80 |
| Average objects per image | 7.06 |
| Mean detection confidence | 0.6458 |
| Median detection confidence | 0.682 |
| Minimum detection confidence | 0.25 |
| Maximum detection confidence | 0.992 |

The most frequently detected class was `person`, with 24,670 detections.

The average of approximately seven detections per image provides an indication of the amount of object-level information available in the metadata. The mean confidence score was approximately 64.58%.

These detections were automatically generated and were not individually verified by human annotators. Therefore, detection confidence should not be interpreted as a guarantee of object-label correctness.

The statistics are stored in:
- `exploration/object_statistics.json`
- `exploration/object_class_statistics.csv`

## 3. Data Exploration Visualizations

Ten visualizations were generated to support the statistical analysis:

1. `fig01_caption_before_after.png` — caption statistics before and after exclusion.
2. `fig02_caption_length_distribution.png` — caption-length distribution.
3. `fig03_object_cooccurrence_matrix.png` — object co-occurrence patterns.
4. `fig04_objects_per_image.png` — distribution of detected objects per image.
5. `fig05_raw_vs_cleaned_captions.png` — comparison of raw and cleaned captions.
6. `fig06_scene_complexity_distribution.png` — distribution of scene complexity.
7. `fig07_top_15_object_classes.png` — the 15 most frequent detected object classes.
8. `fig08_top_50_object_classes.png` — the 50 most frequent detected object classes.
9. `fig09_objects_vs_caption_length.png` — relationship between detected objects and caption length.
10. `fig10_pos_distribution.png` — part-of-speech distribution.

All figures are stored in `exploration/visualizations/`.

These visualizations provide complementary views of caption characteristics, object categories, object co-occurrence, and linguistic patterns.


### 3.1 Supporting Statistical Tables

The exploration analysis also produced the following supporting CSV files:

- `01_caption_exclusion_comparison.csv` — caption exclusion results.
- `02_caption_statistics.csv` — caption statistics.
- `03_object_class_statistics.csv` — detected object class frequencies.
- `04_object_cooccurrence_matrix.csv` — object co-occurrence data.
- `05_raw_vs_cleaned_caption_statistics.csv` — raw and cleaned caption comparison.
- `06_scene_complexity_statistics.csv` — scene complexity statistics.
- `06_scatter_correlation.csv` — correlation between detected objects and caption length.
- `07_spatial_relationships.csv` — spatial relationship statistics.
- `07_pos_distribution.csv` — grammatical category distribution.

These files are stored in `exploration/`. Additional original CSV files are retained in the same directory.

## 4. Repository Organization and Documentation

The repository was organized into directories according to their roles:

- `data/` contains caption data and object metadata.
- `src/` contains data audit, configuration, and split-generation scripts.
- `splits/` contains dataset manifests, identifiers, and exclusion information.
- `embeddings/` contains embedding arrays and knowledge-base mappings.
- `index/` contains FAISS indexes and index configuration.
- `exploration/` contains statistical tables and visualizations.
- `report/` contains audit records and experimental outputs.
- `docs/` contains project documentation and report sections.

The root `README.md` was rewritten to explain the project objectives, repository structure, dataset statistics, installation procedure, execution workflow, and experimental limitations.

A `requirements.txt` file was also created to record the package versions used in the development environment. It includes the principal libraries used for numerical processing, embeddings, retrieval, object detection, generation, and evaluation.

## 5. Reproducibility and Limitations

The exploration outputs were saved as CSV, JSON, and PNG files so that the results can be inspected without rerunning every analysis.

The following limitations should be considered:

1. YOLO detections are automatically generated and have not been manually validated.
2. The caption comparison describes leakage exclusion, not necessarily linguistic text preprocessing.
3. Package versions reflect the development environment; installation may require adjustments for other hardware or operating systems.

Overall, the exploration provides a quantitative description of the caption knowledge base and object metadata, while the repository documentation improves project organization and reproducibility.
