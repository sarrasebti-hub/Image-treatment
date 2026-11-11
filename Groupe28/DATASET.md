# Dataset Details - Group 28

## 1. Data Sources and Quantities
The project uses the **MS COCO 2017** dataset. 

*   **Training (Knowledge Base):** 10,000 images initially selected from the official `train2017` split. After removing exact duplicates (e.g., duplicate file `000000531721.jpg`), the clean training set contains **9,999 images**.
*   **Validation (Hyperparameter Tuning):** 100 images selected from the official `train2017` split.
*   **Test (Final Evaluation):** 1,000 images selected from the official `val2017` split.
*   **Annotations:** We use the official COCO annotations for captions (`captions_*.json`) and object instances (`instances_*.json`) to link images with their ground-truth text and bounding boxes.

## 2. Dataset Splitting (Anti-Leakage)
To ensure a fair evaluation and prevent data leakage:
*   The training and validation sets come from `train2017`.
*   The test set comes strictly from `val2017`.
*   They are stored in strictly separate folders (`data/train`, `data/validation`, `data/test`).
*   The image IDs are frozen into JSON files (`train_ids.json`, `val_ids.json`, `test_ids.json`) to guarantee that no test image is ever retrieved for itself during the RAG pipeline.
*   The original reference captions for the test set are kept separately and are **never** used during the retrieval or generation phases.

## 3. Image Preprocessing (Redimensionnement et Nettoyage)
All 10,099 training/validation images and 1,000 test images went through the following steps:
*   **Integrity Check:** Verified with Pillow to ensure no images were corrupted (all passed).
*   **Duplicate Removal:** SHA-256 hashing was used to detect exact duplicates (1 duplicate removed from training).
*   **EXIF Orientation:** Images were opened and their EXIF orientation was corrected to ensure they are viewed upright.
*   **Colour Conversion (RGB):** All images were converted to RGB mode (even the grayscale ones), ensuring they match the input requirements of the vision encoders (e.g., CLIP).
*   **Grayscale (L mode):** A separate grayscale copy of every image was generated and saved for later experiments.
*   **Saving:** Images were saved as JPEG with a quality of 95. 

## 4. Caption Preprocessing (Nettoyage de Légende)
To preserve the original meaning while preparing the text for retrieval and embedding:
*   **Original Captions:** The original human-written COCO captions are kept completely untouched. These are used for the final evaluation metrics (BLEU, CIDEr, etc.).
*   **Cleaned Captions:** A separate cleaned version was generated for the retrieval and embedding phases. The cleaning steps included:
    *   Removing leading and trailing spaces.
    *   Collapsing repeated whitespace into a single space.
    *   Replacing line breaks with spaces.
    *   *Note: We deliberately did not lowercase the text or remove punctuation to preserve the natural language structure.*
*   **Test Set Captions:** The 1,000 test images have 5,002 original reference captions. After whitespace normalization and removing one within-image duplicate, the cleaned test set contains 5,001 captions.

## 5. Metadata Construction
For every image, a metadata record was constructed linking:
*   Image ID, file name, and paths (RGB and Grayscale).
*   The original and cleaned captions.
*   The COCO object annotations (bounding boxes, category IDs, and category names).
*   The training/validation metadata contains **10,099 records** with **72,998 object annotations**.
*   The test metadata contains **1,000 records** with **7,375 object annotations**.
