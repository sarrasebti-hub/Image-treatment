# Statistical Exploration of the MS COCO Dataset

## 1. Overview

This directory contains the statistical exploration of the selected MS COCO dataset subset.

The purpose of this stage is to understand the dataset's visual and textual characteristics before proceeding with further preprocessing, embedding generation, and knowledge-base construction.

The analysis results will also provide documented evidence about the dataset's composition, complexity, and diversity for the final project report.

## 2. Objectives

The statistical exploration aims to:

- Examine the distribution and frequency of annotated object classes.
- Analyse scene complexity by studying the number of objects per image.
- Investigate the distribution of spatial relationships, where suitable annotation information is available.
- Analyse caption length and vocabulary characteristics.
- Examine part-of-speech (POS) distributions.
- Investigate visual attributes, including colours and shapes.
- Explore scene contexts, such as indoor/outdoor and day/night, where suitable information can be obtained.
- Produce statistical tables and visualisations to support the findings.

## 3. Analysis Scope

The exploration follows the requirements specified in Section 4.3 of the project specification.

### 3.1 Object Class Distribution

Analyse the frequency of annotated object categories using COCO instance annotations.

Expected outputs:

- Object-class frequency table.
- Top 50 most frequent object classes.
- Visualisation of object-class distribution.

### 3.2 Scene Complexity

Study the number of annotated objects present in individual images.

Expected outputs:

- Average number of objects per image.
- Minimum, maximum, median, and standard deviation.
- Distribution of object counts across images.
- Relevant visualisations, such as histograms.

### 3.3 Spatial Relationships

Investigate spatial relationships between objects where suitable information is available.

Expected outputs:

- Relationship statistics, if supported by available data.
- Relevant relationship visualisations.

### 3.4 Textual Statistics

Examine the human-written reference captions associated with the selected images.

Expected outputs:

- Average caption length in words or tokens.
- Caption-length distribution.
- Vocabulary size.
- Part-of-speech distribution.

Reference captions must remain unchanged for evaluation purposes.

### 3.5 Diversity Analysis

Investigate the diversity of visual attributes and scene contexts.

The intended analysis includes:

- Colours.
- Shapes.
- Indoor and outdoor contexts.
- Day and night contexts.

The availability and reliability of the required information will be examined before selecting the appropriate analysis method.

### 3.6 Data Visualisation

Use suitable visualisations to communicate the statistical findings.

Examples include:

- Histograms.
- Bar charts.
- Scatter plots.
- Object co-occurrence matrices.
- Other relevant statistical plots.

## 4. Directory Structure

```text
01_statistical_exploration/
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── statistics/
│
├── report/
│   └── statistical_exploration.md
│
└── README.md
```

The analysis notebooks are maintained separately under:

```text
01_src/notebooks/
```

### Output directory descriptions

| Directory | Content |
|---|---|
| `outputs/tables/` | Exported statistical tables, preferably CSV files. |
| `outputs/figures/` | Generated charts and visualisations, preferably PNG files. |
| `outputs/statistics/` | Structured statistical summaries, preferably JSON files. |
| `report/` | Written observations, interpretations, and conclusions. |

## 5. Input Data

The analysis uses the processed dataset located in:

```text
02_processed_data/
```

The three available splits are:

- `training/`
- `validation/`
- `test/`

The analysis must preserve the separation between the splits.

Training annotations can be used to study the knowledge-base dataset. Validation and test annotations may be analysed separately for descriptive purposes, but their reference captions must never be introduced into the retrieval knowledge base.

The test split remains reserved for final evaluation.

## 6. Analysis Environment

The analysis is implemented using Python and Jupyter Notebook.

Potential libraries include:

- NumPy: numerical operations.
- Pandas: data manipulation and statistical summaries.
- Matplotlib: data visualisation.
- Seaborn: statistical visualisation.
- NLTK or spaCy: textual analysis and POS tagging.
- JSON and pathlib: annotation loading and file management.

The exact libraries and versions will be documented once the environment is configured.

## 7. Reproducibility and Output Management

Each analysis should be reproducible from the corresponding notebook.

The following practices will be followed:

- Use the processed dataset rather than modifying the original COCO files.
- Keep analysis code separate from generated outputs.
- Save important statistical results to files.
- Export relevant visualisations for use in the final report.
- Document assumptions, limitations, and processing decisions.
- Avoid manually changing generated statistical results.
- Keep the written interpretation consistent with the saved outputs.

## 8. Deliverables

At the end of this stage, the directory should contain:

1. Reproducible Jupyter notebook(s).
2. Statistical tables.
3. Generated visualisations.
4. Structured statistical summaries.
5. A written document explaining the methodology, observations, and findings.

These deliverables will contribute to the dataset description and statistical exploration sections of the final project report.

## 9. Limitations

Some requested characteristics, such as spatial relationships, shapes, and scene contexts, may not be directly represented in the available COCO annotations.

Their analysis will therefore depend on the availability of suitable metadata or additional processing methods.

Any inferred or automatically detected information must be distinguished from original human annotations and documented accordingly.

