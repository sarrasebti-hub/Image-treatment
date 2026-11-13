import numpy as np
from pathlib import Path

folder = Path("data/embeddings")

files = [
    "image_embeddings.npy",
    "image_ids.npy",
    "text_embeddings.npy",
    "text_image_ids.npy",
    "text_annotation_ids.npy"
]

for filename in files:
    data = np.load(folder / filename)

    print(f"\n{filename}")
    print("Shape:", data.shape)
    print("Data type:", data.dtype)

print("\nEMBEDDINGS VERIFICATION COMPLETED")