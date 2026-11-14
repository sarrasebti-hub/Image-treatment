PHASE 1 - KNOWLEDGE BASE AND RETRIEVAL ARTIFACTS

Project:
A comparative study of RAG techniques for generating image captions

Image and text encoder:
openai/clip-vit-base-patch32

Embedding dimension:
512

Knowledge base:
10,000 COCO train2017 images
5 captions per image
50,000 caption vectors

Vector database:
FAISS

Image index:
IndexFlatIP - 10,000 vectors

Text index:
IndexFlatIP - 50,000 vectors

Similarity:
Inner Product (equivalent to cosine similarity
for L2-normalized vectors)

Metadata:
COCO object information, object categories,
spatial relationships, and caption-derived mentions.

Quality control:
Final index QC report: 19/19 checks passed.

IMPORTANT:
The original and preprocessed image files are not included
in this package. They remain in the separate dataset storage.

The dataset split files and statistical exploration outputs
are also maintained separately in the local project.

This package contains the knowledge-base records,
embeddings, FAISS indexes, mappings, metadata and reports.