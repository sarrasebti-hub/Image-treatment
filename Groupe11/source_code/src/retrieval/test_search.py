import json
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
import torch

print("Loading FAISS index and CLIP...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Load the FAISS index and data
index = faiss.read_index("outputs/faiss_index.bin")
with open("outputs/valid_kb_ids.json", "r") as f:
    valid_ids = json.load(f)
with open("outputs/captions.json", "r") as f:
    captions_data = json.load(f)

# Create a dictionary to look up captions by image ID
caption_dict = {}
for item in captions_data:
    img_id = item['image_id']
    if img_id not in caption_dict:
        caption_dict[img_id] = []
    # Use Membre 1's exact key
    caption_dict[img_id].append(item['caption_original'])

# Our Search query
query = "a black dog running on the grass"
print(f"\nSearching for: '{query}'")

# Convert query to a vector
text_inputs = processor(text=[query], return_tensors="pt", padding=True)
with torch.no_grad():
    features = model.get_text_features(**text_inputs).pooler_output
    query_vec = features.numpy().flatten()
    query_vec = query_vec / np.linalg.norm(query_vec)
    query_vec = np.array([query_vec]).astype('float32')

# Search FAISS for the top 3 results
k = 3
distances, indices = index.search(query_vec, k)

print("\n--- Top Results ---")
for i in range(k):
    idx = indices[0][i]
    score = distances[0][i]
    img_id = valid_ids[idx]
    
    print(f"\nResult {i+1} (Score: {score:.4f}) - Image ID: {img_id}")
    if img_id in caption_dict:
        for cap in caption_dict[img_id]:
            print(f"  - {cap}")