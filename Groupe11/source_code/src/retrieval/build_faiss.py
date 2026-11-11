import os
import json
import time
import torch
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

print("Loading CLIP model...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

with open("outputs/splits.json", "r") as f:
    splits = json.load(f)
with open("outputs/captions.json", "r") as f:
    captions_data = json.load(f)

kb_ids = splits["kb_ids"]
image_dir = "data/raw/images"
output_dir = "outputs"

# --- 1. IMAGE EMBEDDINGS ---
image_emb_path = os.path.join(output_dir, "image_embeddings.npy")
ids_path = os.path.join(output_dir, "valid_kb_ids.json")

if os.path.exists(image_emb_path) and os.path.exists(ids_path):
    print("✅ Image embeddings and IDs already exist. Loading them...")
    embeddings_matrix = np.load(image_emb_path)
    with open(ids_path, "r") as f:
        valid_ids = json.load(f)
else:
    print(f"Generating image embeddings for {len(kb_ids)} images...")
    embeddings = []
    valid_ids = []
    with torch.no_grad():
        for img_id in tqdm(kb_ids):
            img_id_str = str(img_id).zfill(12)
            img_path = os.path.join(image_dir, f"{img_id_str}.jpg")
            if not os.path.exists(img_path): 
                continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            features = model.get_image_features(**inputs).pooler_output
            vec = features.numpy().flatten()
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
            valid_ids.append(img_id)
            
    embeddings_matrix = np.array(embeddings).astype('float32')
    np.save(image_emb_path, embeddings_matrix)
    with open(ids_path, "w") as f:
        json.dump(valid_ids, f)

# --- 2. TEXT EMBEDDINGS ---
text_emb_path = os.path.join(output_dir, "text_embeddings.npy")
if os.path.exists(text_emb_path) and os.path.getsize(text_emb_path) > 1000: # Check if file is not empty
    print("✅ Text embeddings already exist. Loading them...")
    text_embeddings_matrix = np.load(text_emb_path)
else:
    print(f"\nGenerating text embeddings...")
    text_embeddings = []
    valid_ids_set = set(valid_ids) 
    
    with torch.no_grad():
        for item in tqdm(captions_data):
            img_id = item.get('image_id')
            if img_id not in valid_ids_set: 
                continue
            
            # Use Membre 1's exact key: 'caption_original'
            cap = item.get('caption_original')
            if not cap:
                continue
                
            text_inputs = processor(text=[cap], return_tensors="pt", padding=True, truncation=True)
            features = model.get_text_features(**text_inputs).pooler_output
            vec = features.numpy().flatten()
            vec = vec / np.linalg.norm(vec)
            text_embeddings.append(vec)
                
    text_embeddings_matrix = np.array(text_embeddings).astype('float32')
    np.save(text_emb_path, text_embeddings_matrix)

# --- 3. BUILD FAISS INDEX ---
print(f"\nBuilding FAISS index...")
start_time = time.time()
dimension = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_matrix)
index_time = time.time() - start_time
faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))

# --- 4. DOCUMENTATION ---
doc_text = f"""# FAISS Index Documentation (Membre 2)
- **Index Type:** Flat Inner Product (Equivalent to Cosine Similarity)
- **Embedding Dimension:** {dimension}
- **Total Image Vectors Indexed:** {len(embeddings_matrix)}
- **Total Text Vectors Generated:** {len(text_embeddings_matrix)}
- **Indexing Time:** {index_time:.2f} seconds
- **Model Used:** {model_name}
"""
with open(os.path.join(output_dir, "index_documentation.md"), "w") as f:
    f.write(doc_text)

print("✅ DONE! Check the outputs/ folder for .npy files, FAISS index, and documentation.")