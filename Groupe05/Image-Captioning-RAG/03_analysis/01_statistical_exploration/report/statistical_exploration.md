# Statistical Exploration of the MS-COCO Dataset

## 1. Introduction

Statistical exploration is an important step in understanding the characteristics, structure, and diversity of a dataset before developing an image-captioning system. It provides an overview of the visual content represented by the images and the linguistic characteristics of their associated captions.

This analysis was conducted on the training subset of the MS-COCO dataset prepared for the image-captioning project. It examines object-category distribution, scene complexity, caption statistics, linguistic characteristics, diversity indicators, and object co-occurrence patterns.

The main objectives are to:

- Identify the most frequent object categories in the dataset.
- Examine the distribution of the number of annotated objects per image.
- Analyse caption length, vocabulary size, and part-of-speech distribution.
- Explore the frequency of colour, shape, and contextual terms in captions.
- Identify recurring combinations of object categories within images.
- Produce statistical tables and visualisations that help describe the dataset.

The findings provide an initial understanding of the dataset and help identify characteristics that may influence image-caption generation and retrieval.

## 2. Dataset Description

### 2.1 Dataset overview

The analysis uses a subset of the MS-COCO 2017 dataset. The original dataset provides images, human-written captions, and object annotations.

The training subset analysed in this report contains:

| Metric | Value |
|---|---:|
| Training images | 9,900 |
| Captions per image | 5 |
| Total captions | 49,500 |
| Object annotations | 71,326 |
| Object categories defined in COCO | 80 |

Each selected training image is associated with five reference captions. Object annotations are provided separately and include information such as image ID, category ID, bounding box, segmentation, area, and crowd indicator.

### 2.2 Data sources

Two principal annotation files were used:

- **Captions annotations:** provide the image IDs and their corresponding human-written captions.
- **Instances annotations:** provide object-level annotations, including category IDs and bounding-box information.

The COCO category definitions were used to map category IDs to their corresponding human-readable category names.

### 2.3 Analysis scope

The statistical exploration focuses on the training subset. Validation and test reference captions were not included in these calculations.

The results therefore describe the selected training data and should not automatically be interpreted as statistics for the complete MS-COCO dataset.

## 3. Object Class Distribution

### 3.1 Methodology

The object-class distribution was calculated using the instances annotations.

Each object annotation was associated with a category ID. The category IDs were mapped to their corresponding names, and the frequency of each category was calculated across the training annotations.

The categories were then sorted in descending order of frequency. The 50 most frequent categories were selected for detailed examination.

The frequency percentage was calculated as:

\[
P(c)=\frac{N_c}{N_{\text{objects}}}\times 100
\]

where:

- \(N_c\) represents the number of annotations belonging to category \(c\).
- \(N_{\text{objects}}\) represents the total number of object annotations.

### 3.2 Results

The analysis identified 71,326 object annotations across the 9,900 training images, using the 80 category definitions provided by COCO.

The most frequent category is **person**, with 21,658 annotations, representing approximately 30.36% of all object annotations.

Other frequently occurring categories include car (3,807), chair (3,153), book (2,166), bottle (1,858), and cup (1,622).

The following table presents the 20 most frequent categories.

| Rank | Category | Frequency | Percentage |
|---:|---|---:|---:|
| 1 | Person | 21,658 | 30.36% |
| 2 | Car | 3,807 | 5.34% |
| 3 | Chair | 3,153 | 4.42% |
| 4 | Book | 2,166 | 3.04% |
| 5 | Bottle | 1,858 | 2.60% |
| 6 | Cup | 1,622 | 2.27% |
| 7 | Dining table | 1,278 | 1.79% |
| 8 | Traffic light | 1,123 | 1.57% |
| 9 | Handbag | 1,013 | 1.42% |
| 10 | Bowl | 1,012 | 1.42% |
| 11 | Umbrella | 1,002 | 1.40% |
| 12 | Bird | 966 | 1.35% |
| 13 | Motorcycle | 872 | 1.22% |
| 14 | Truck | 862 | 1.21% |
| 15 | Sheep | 844 | 1.18% |
| 16 | Banana | 833 | 1.17% |
| 17 | Boat | 810 | 1.14% |
| 18 | Bench | 807 | 1.13% |
| 19 | Potted plant | 756 | 1.06% |
| 20 | Kite | 729 | 1.02% |

**Figure 1.** Frequency distribution of the 50 most frequent object categories.

*File: `outputs/figures/top_50_object_classes.png`*

### 3.3 Discussion

The distribution indicates a considerable imbalance between object categories. The person category alone accounts for approximately 30.36% of all object annotations, making it substantially more frequent than any other category.

Objects associated with everyday environments, such as chairs, books, bottles, cups, and dining tables, are also relatively common. Vehicles and outdoor-related categories, including cars, traffic lights, motorcycles, trucks, and boats, are represented as well.

This distribution illustrates the variety of visual content in the dataset, but also reveals that some categories are much more strongly represented than others. Such imbalance should be considered when analysing retrieval results and evaluating the ability of a captioning system to describe less frequent objects.

## 4. Scene Complexity Analysis

### 4.1 Methodology

Scene complexity was examined by calculating the number of annotated object instances in each training image.

For every image, the number of associated object annotations was counted. Images without any object annotations were also retained in the analysis and assigned an object count of zero.

The following statistical measures were calculated:

- Mean number of objects per image.
- Median number of objects per image.
- Standard deviation.
- Minimum and maximum object counts.
- Distribution of images according to their object counts.

### 4.2 Results

The analysis produced the following results:

| Metric | Value |
|---|---:|
| Number of images | 9,900 |
| Total annotated objects | 71,326 |
| Average objects per image | 7.20 |
| Median objects per image | 4 |
| Standard deviation | 7.27 |
| Minimum objects per image | 0 |
| Maximum objects per image | 65 |

The object-count distribution also shows that:

- 86 images contain zero annotated objects (0.87%).
- 1,100 images contain one annotated object (11.11%).
- 1,854 images contain two annotated objects (18.73%).

**Figure 2.** Distribution of the number of annotated objects per image.

*File: `outputs/figures/objects_per_image_distribution.png`*

### 4.3 Discussion

The average number of annotated objects per image is 7.20, while the median is 4. This difference indicates that the distribution is not evenly balanced: images containing many objects increase the average.

The standard deviation of 7.27 also indicates substantial variation in scene complexity. Some images contain only a small number of annotated objects, while others contain considerably more. The maximum observed count is 65 objects in a single image.

These findings demonstrate that the dataset includes images with different levels of visual complexity. This is relevant to image captioning because describing a simple scene may require identifying only a few important objects, whereas describing a complex scene may require selecting the most relevant objects and relationships without producing an excessively detailed caption.

It should be noted that object counts are based on available COCO annotations. They do not necessarily represent every object visible in an image.

### 4.4 Spatial relationship information

The inspected instance annotations contain bounding boxes, segmentation information, object categories, and other instance-level fields. However, they do not provide explicit semantic relationship labels such as *person riding bicycle* or *cup on table*.

Consequently, this analysis does not claim to measure semantic spatial relationships directly. The object co-occurrence analysis presented later identifies categories appearing together in the same image, which is a different type of information.

## 5. Textual Statistics

### 5.1 Caption length analysis

The caption-length analysis examines the number of words used in the human-written descriptions.

For each caption, the number of words was calculated using whitespace-based splitting. The resulting values were used to calculate the mean, median, standard deviation, minimum, and maximum caption lengths.

The analysis covers all 49,500 training captions.

#### Results

| Metric | Value |
|---|---:|
| Total captions | 49,500 |
| Average caption length | 10.46 words |
| Median caption length | 10 words |
| Standard deviation | 2.39 words |
| Minimum caption length | 6 words |
| Maximum caption length | 44 words |

**Figure 3.** Distribution of caption lengths.

*File: `outputs/figures/caption_length_distribution.png`*

#### Discussion

The average caption length is approximately 10.46 words, while the median is 10 words. This indicates that typical captions are relatively short and concise.

The maximum length of 44 words demonstrates that some captions are considerably longer than the typical description. These longer captions may contain additional details about objects, actions, or scene context.

Overall, the results suggest that the dataset primarily uses short natural-language descriptions, although some variation exists in the level of descriptive detail.

The word-count method used here is a simple whitespace-based measure rather than a linguistic tokenizer.

### 5.2 Vocabulary analysis

Vocabulary analysis was performed to measure the lexical diversity of the captions.

All captions were converted to lowercase, and words were extracted using a regular expression that retains alphabetic English words. The extracted words were then counted to determine the total number of word occurrences and the number of distinct words.

#### Results

| Metric | Value |
|---|---:|
| Total word occurrences | 518,774 |
| Vocabulary size | 9,514 |
| Average frequency per distinct word | 54.53 |
| Most frequent word | a |
| Frequency of the most frequent word | 81,950 |

The 30 most frequent words were also extracted and visualised.

**Figure 4.** Thirty most frequent words in the training captions.

*File: `outputs/figures/top_30_frequent_words.png`*

#### Discussion

The vocabulary contains 9,514 distinct normalized words across 518,774 word occurrences.

The most frequent word is *a*, followed by common grammatical words such as *on*, *of*, *the*, *in*, and *with*. These words are frequently used to construct natural-language descriptions.

Content-related words such as *man*, *sitting*, *people*, *standing*, *woman*, and *table* also appear frequently. This reflects the descriptive nature of the captions, which commonly refer to people, objects, and actions.

The vocabulary size provides an indication of lexical diversity. However, it should be interpreted in the context of the normalization method: lowercasing and alphabetic word extraction mean that different original forms may be treated as the same word or excluded.

### 5.3 Part-of-speech distribution

Part-of-speech (POS) analysis was conducted using the spaCy English language model `en_core_web_sm`.

The captions were processed to identify grammatical categories, including nouns, verbs, adjectives, determiners, adpositions, and other parts of speech.

The frequency and percentage of each POS category were calculated.

#### Results

| POS category | Frequency | Percentage |
|---|---:|---:|
| NOUN | 179,165 | 34.53% |
| DET | 106,217 | 20.47% |
| ADP | 87,364 | 16.84% |
| VERB | 50,435 | 9.72% |
| ADJ | 37,191 | 7.17% |
| AUX | 12,222 | 2.36% |
| CCONJ | 12,178 | 2.35% |
| PRON | 11,551 | 2.23% |
| ADV | 7,420 | 1.43% |
| NUM | 6,325 | 1.22% |
| PROPN | 4,548 | 0.88% |
| PART | 2,156 | 0.42% |
| SCONJ | 2,025 | 0.39% |
| X | 62 | 0.01% |
| INTJ | 45 | 0.01% |
| PUNCT | 3 | Less than 0.01% |

**Figure 5.** Part-of-speech distribution across the captions.

*File: `outputs/figures/pos_distribution.png`*

#### Discussion

Nouns represent the largest POS category, accounting for approximately 34.53% of the analysed tokens. This is expected in image descriptions because captions frequently identify the people, animals, and objects present in a scene.

Determiners and adpositions are also common, representing 20.47% and 16.84%, respectively. Verbs account for 9.72%, while adjectives represent 7.17%.

The presence of nouns, verbs, and adjectives is particularly relevant to image captioning: nouns identify visual entities, verbs describe actions, and adjectives provide additional descriptive information.

The results demonstrate that the captions use a combination of object references, actions, and grammatical structures to describe images.

## 6. Diversity Analysis

### 6.1 Methodology

Diversity analysis was performed to investigate how frequently captions refer to visual attributes and scene contexts.

The analysis used predefined lists of English words related to:

- Colours.
- Shapes.
- Indoor and outdoor environments.
- Daytime and nighttime contexts.

Captions were converted to lowercase and tokenized. The frequency of each selected word was then calculated.

Two measurements were considered:

- **Word frequency:** the total number of occurrences of a selected word.
- **Caption frequency:** the number of captions containing that word.

This distinction is important because a word may appear more than once in a caption.

The analysis is based on textual mentions in the captions, not on direct visual classification of the images.

### 6.2 Colour attributes

The selected colour vocabulary included red, blue, green, yellow, black, white, orange, pink, purple, brown, gray, and grey.

#### Results

| Colour | Word frequency | Captions mentioning colour |
|---|---:|---:|
| White | 2,999 | 2,894 |
| Black | 1,704 | 1,675 |
| Red | 1,538 | 1,513 |
| Green | 1,125 | 1,109 |
| Blue | 1,109 | 1,084 |
| Yellow | 734 | 726 |
| Brown | 713 | 701 |
| Orange | 417 | 410 |
| Pink | 255 | 250 |
| Gray | 158 | 156 |
| Purple | 133 | 131 |
| Grey | 122 | 122 |

**Figure 6.** Frequency of colour mentions in the captions.

*File: `outputs/figures/colour_mentions.png`*

#### Discussion

White is the most frequently mentioned colour, followed by black and red. Green and blue also appear relatively frequently.

The results indicate that colour descriptions are present in the captions, although they are not equally distributed. Some colours are mentioned much less frequently than others.

Gray and grey are alternative spellings of the same colour and should be considered together when discussing the overall frequency of this attribute.

### 6.3 Shape attributes

The selected shape vocabulary included round, square, rectangular, circular, oval, triangular, curved, and straight.

#### Results

| Shape | Word frequency | Captions mentioning shape |
|---|---:|---:|
| Round | 72 | 72 |
| Square | 52 | 52 |
| Circular | 17 | 17 |
| Straight | 15 | 15 |
| Rectangular | 11 | 11 |
| Curved | 9 | 9 |
| Oval | 3 | 3 |
| Triangular | 2 | 2 |

**Figure 7.** Frequency of shape mentions in the captions.

*File: `outputs/figures/shape_mentions.png`*

#### Discussion

Shape-related terms occur much less frequently than colour-related terms. The most frequent shape description is *round*, followed by *square*.

This suggests that explicit shape descriptions are relatively uncommon in the selected captions. However, this result should not be interpreted as evidence that the images contain few objects with these shapes. It only indicates that the selected shape vocabulary is rarely used in the human-written descriptions.

### 6.4 Scene context

Contextual diversity was explored using four groups of terms:

- Indoor: indoor, inside, room, kitchen, bedroom, office, and bathroom.
- Outdoor: outdoor, outside, street, park, field, beach, and garden.
- Day: day, daytime, sunny, and sunlight.
- Night: night, nighttime, dark, and evening.

#### Results

| Context | Word frequency | Captions mentioning context |
|---|---:|---:|
| Indoor | 4,551 | 4,328 |
| Outdoor | 6,557 | 6,365 |
| Day | 516 | 397 |
| Night | 350 | 343 |

**Figure 8.** Frequency of contextual mentions in the captions.

*File: `outputs/figures/context_mentions.png`*

#### Discussion

Outdoor-related terms occur more frequently than indoor-related terms in the analysed captions. There are 6,365 captions containing at least one selected outdoor term, compared with 4,328 captions containing at least one selected indoor term.

Explicit references to daytime and nighttime are considerably less frequent. This suggests that captions do not always describe lighting conditions or the time of day, even when such information may be visually present.

These measurements are lexical indicators rather than definitive scene classifications. The predefined word lists are not exhaustive, and a caption may contain terms from more than one context group. Therefore, the percentages or counts should not be interpreted as mutually exclusive image categories.

## 7. Object Co-occurrence Analysis

### 7.1 Methodology

Object co-occurrence analysis was conducted to identify object categories that frequently appear together in the same image.

For each image, the set of distinct annotated object categories was extracted. Each possible pair of different categories appearing in that image contributed one count to the co-occurrence matrix.

The matrix was constructed across the 80 COCO object categories.

An entry \(C_{ij}\) represents the number of images in which both category \(i\) and category \(j\) are present.

The diagonal entries were set to zero because the analysis considers pairs of different categories. The matrix is symmetrical because the co-occurrence of category A with category B is equivalent to the co-occurrence of category B with category A.

### 7.2 Results

The analysis identified 9,814 images containing at least one annotated object. The remaining 86 images had no object annotations.

The strongest co-occurring category pairs include:

| Category A | Category B | Co-occurrence frequency |
|---|---|---:|
| Person | Car | 719 |
| Person | Chair | 641 |
| Person | Handbag | 516 |
| Person | Dining table | 465 |
| Chair | Dining table | 441 |
| Person | Backpack | 405 |
| Person | Cup | 393 |
| Cup | Dining table | 386 |
| Person | Bottle | 361 |
| Person | Bench | 352 |
| Person | Sports ball | 350 |
| Person | Truck | 338 |
| Person | Cell phone | 326 |
| Car | Truck | 313 |
| Person | Umbrella | 296 |

**Figure 9.** Object co-occurrence heatmap for the 20 most frequent object categories.

*File: `outputs/figures/object_cooccurrence_heatmap.png`*

The complete co-occurrence matrix and ranked category-pair results are available in the corresponding CSV files.

### 7.3 Discussion

The results show that the person category has the strongest co-occurrence with many other categories. The most frequent pair is person–car, appearing in 719 images, followed by person–chair (641) and person–handbag (516).

Several combinations involving furniture and dining objects are also prominent, including chair–dining table (441), cup–dining table (386), and bottle–cup (242).

These combinations may reflect recurring visual compositions in the dataset, such as people in everyday environments, dining scenes, and outdoor or urban settings.

The co-occurrence matrix also reveals relationships between object categories that do not necessarily involve people. For example, car–truck appears in 313 images, while bottle–cup appears in 242 images.

It is important to distinguish object co-occurrence from semantic or spatial relationships. Two objects appearing in the same image does not necessarily mean that they interact or have a specific spatial arrangement. For example, the person–car pair does not establish that the person is driving the car.

## 8. General Discussion

The statistical exploration provides several observations about the selected MS-COCO training subset.

First, the object-category distribution is highly imbalanced. The person category dominates the object annotations, while many other categories occur considerably less frequently.

Second, scene complexity varies substantially. Although the average image contains 7.20 annotated objects, the median is only four, indicating that some images with many objects increase the overall average.

Third, the captions are generally concise. Their average length is approximately 10.46 words, and the vocabulary contains 9,514 distinct normalized words. The POS analysis shows that nouns are the most frequent grammatical category, followed by determiners and adpositions.

Fourth, the diversity analysis indicates that colour terms are more frequently mentioned than shape terms. Outdoor-related vocabulary is also more frequent than indoor-related vocabulary in the selected captions. These results describe the language used in the captions rather than definitive visual properties of the images.

Finally, the co-occurrence analysis reveals recurring combinations of object categories, particularly those involving people, vehicles, furniture, and dining objects. These patterns provide useful information about the composition of the dataset.

Taken together, the findings demonstrate that the dataset contains diverse visual scenes and natural-language descriptions, but also exhibits imbalances in object frequency, scene complexity, and descriptive vocabulary.

## 9. Limitations

Several limitations should be considered when interpreting the results.

1. **Subset limitation:** The analysis was conducted on 9,900 training images rather than the complete MS-COCO dataset. The results describe this selected subset only.

2. **Annotation limitation:** Object counts depend on the available COCO instance annotations. They may not include every visible object in an image.

3. **Spatial relationship limitation:** The inspected instance annotations provide bounding boxes and object categories but do not contain explicit semantic relationship labels. Object co-occurrence cannot be treated as a direct measure of semantic interaction.

4. **Vocabulary limitation:** The vocabulary size depends on the text-normalization and token-extraction method. Lowercasing and alphabetic extraction may combine or exclude certain word forms.

5. **POS tagging limitation:** Part-of-speech labels are generated automatically using a language model and may contain classification errors.

6. **Diversity analysis limitation:** Colour, shape, and context measurements rely on predefined word lists. They do not identify every possible expression or directly classify image content.

7. **Co-occurrence limitation:** Co-occurrence counts indicate that two categories appear in the same image, but do not establish their spatial arrangement or interaction.

These limitations should be considered in subsequent experiments and when interpreting the performance of image-captioning and retrieval systems.

## 10. Conclusion

This statistical exploration examined the visual and linguistic characteristics of the selected MS-COCO training subset, containing 9,900 images and 49,500 captions.

The analysis covered object-category distribution, scene complexity, caption length, vocabulary size, part-of-speech distribution, diversity indicators, and object co-occurrence.

The results revealed a strong presence of people in the object annotations, substantial variation in the number of objects per image, relatively short captions, and a vocabulary containing 9,514 distinct normalized words. Nouns were the dominant grammatical category, while colour terms appeared more frequently than shape-related terms. The co-occurrence analysis further identified recurring combinations of object categories, particularly those involving people, vehicles, furniture, and dining objects.

These findings establish a statistical baseline for understanding the dataset before proceeding with the subsequent stages of the project. They also provide useful context for the development of the knowledge base and the evaluation of retrieval-augmented image-captioning techniques.