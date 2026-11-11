"""Yahia's task: embeddings, vector database and quality control.

Modules
-------
config            Project paths and shared constants.
data              Split loading, image validation and COCO image download.
encoder           CLIP image/text encoder (shared embedding space).
build_embeddings  Encode a split and persist .npy + JSONL artifacts.
build_vector_db   Build the FAISS vector database from the embeddings.
search            Text -> image, image -> text and image -> image search.
quality_control   Automated quality-control checks and Markdown report.
pipeline          One-command orchestration of the whole task.
"""

__version__ = "1.0.0"
