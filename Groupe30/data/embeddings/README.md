# Embeddings

Only the manifest is stored here. The embedding files are on the team's Google Drive:
`kb_image_embeddings.npy` (10,000 x 512), `kb_caption_embeddings.npy` (50,031 x 512), `val_image_embeddings.npy` (100 x 512), `test_image_embeddings.npy` (1,000 x 512).

All vectors come from CLIP `openai/clip-vit-base-patch32`, 512 dimensions, float32, L2-normalised. See `embedding_manifest.json`.
