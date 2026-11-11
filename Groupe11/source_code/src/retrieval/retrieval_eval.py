import time
import json
import torch
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

print("Loading model and FAISS...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
index = faiss.read_index("outputs/faiss_index.bin")

with open("outputs/valid_kb_ids.json", "r") as f:
    valid_ids = json.load(f)

# We will test latency and Top-K on 5 random images
test_images = valid_ids[:5] 
k_values = [1, 3, 5, 10]
results = []

print("Running Top-K Evaluation...")
for img_id in test_images:
    # 1. Load image
    img_path = f"data/raw/images/{str(img_id).zfill(12)}.jpg"
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # 2. Create Query Vector
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
        query_vec = features.numpy().flatten()
        query_vec = query_vec / np.linalg.norm(query_vec)
        query_vec = np.array([query_vec]).astype('float32')

    # 3. Measure Latency and Retrieve
    start_time = time.time()
    distances, indices = index.search(query_vec, max(k_values))
    latency_ms = (time.time() - start_time) * 1000
    
    # 4. Calculate if the actual image was found in the top K
    for k in k_values:
        top_k_indices = indices[0][:k]
        retrieved_ids = [valid_ids[idx] for idx in top_k_indices]
        
        # Recall@k: Is the query image itself in the top K? (Oracle test)
        hit = 1 if img_id in retrieved_ids else 0
        
        results.append({
            "Image_ID": img_id,
            "K": k,
            "Hit": hit,
            "Latency_ms": round(latency_ms, 2)
        })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("outputs/retrieval_metrics.csv", index=False)
print("✅ DONE! Saved to outputs/retrieval_metrics.csv")
print("\n--- Retrieval Metrics ---")
print(df.groupby("K")["Hit"].mean()) # This is Recall@k