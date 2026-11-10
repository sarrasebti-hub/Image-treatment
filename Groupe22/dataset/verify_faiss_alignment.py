import json
import faiss
import numpy as np

base = "dataset"

index_path = f"{base}/embeddings/train2017.faiss"
ids_path = f"{base}/embeddings/train2017_ids.json"
embeddings_path = f"{base}/dataset/embeddings/image_embeddings.npy"
source_ids_path = f"{base}/dataset/embeddings/image_ids.json"

index = faiss.read_index(index_path)

with open(ids_path, encoding="utf-8") as f:
    faiss_ids = json.load(f)

with open(source_ids_path, encoding="utf-8") as f:
    source_ids = json.load(f)

embeddings = np.load(embeddings_path)

print("========== FAISS ALIGNMENT AUDIT ==========")

print("Index type:", type(index).__name__)
print("Index dimension:", index.d)
print("Vectors in index:", index.ntotal)
print("FAISS IDs:", len(faiss_ids))
print("Source IDs:", len(source_ids))

print("\n========== ID ALIGNMENT ==========")

print("Same number:", len(faiss_ids) == len(source_ids))
print(
    "Same order:",
    [int(x) for x in faiss_ids] ==
    [int(x) for x in source_ids]
)

print("Unique FAISS IDs:", len(set(faiss_ids)))
print("Duplicate FAISS IDs:", len(faiss_ids) - len(set(faiss_ids)))

print("\n========== VECTOR ALIGNMENT ==========")

reconstructed = index.reconstruct_n(0, index.ntotal)

print("Reconstructed shape:", reconstructed.shape)
print("Source shape:", embeddings.shape)

print(
    "Maximum absolute difference:",
    np.max(np.abs(reconstructed - embeddings))
)

print(
    "Vectors identical within tolerance:",
    np.allclose(reconstructed, embeddings, atol=1e-5)
)