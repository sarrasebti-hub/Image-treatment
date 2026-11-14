# Hybrid RAG Image Captioning

## Project Description

Hybrid RAG is an image captioning system combining computer vision, information retrieval, and natural language generation.

## Technologies

- YOLO: Object detection
- CLIP: Image feature extraction
- FAISS: Dense retrieval
- BM25: Sparse retrieval
- RRF: Hybrid retrieval
- BLIP: Baseline caption generation
- FLAN-T5: Final caption generation
- Streamlit: User interface

## Project Structure

- app.py: Main Streamlit application
- preprocessing/: Dataset preprocessing
- object_extraction/: YOLO object detection
- feature_extraction/: CLIP feature extraction
- retrieval/: Dense, sparse and hybrid retrieval
- generation/: Caption generation
- evaluation/: Model evaluation
- scripts/: Utility scripts

## Installation

1. Clone the repository:
   git clone YOUR_GITHUB_REPOSITORY_URL
2. Enter the project folder:
   cd Hybrid_RAG
3. Create a virtual environment:
   python -m venv .venv
4. Activate it:
   .\.venv\Scripts\Activate.ps1
5. Install PyTorch CPU:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
6. Install dependencies:
   pip install -r requirements.txt

## Run the application

streamlit run app.py

## Dataset and Generated Files

Large datasets, COCO annotations, model weights, generated features and retrieval indexes are excluded from this repository. They must be obtained separately or generated using the project scripts.

## Notes

- Python 3.11 is recommended.
- Internet access may be required to download pretrained models.
- CPU execution is supported, but some operations may take longer.

