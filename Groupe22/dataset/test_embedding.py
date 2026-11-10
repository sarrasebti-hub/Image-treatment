
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Chemins
MODEL_NAME = "openai/clip-vit-base-patch32"
METADATA_PATH = "metadata/knowledge_base_metadata.json"
IMAGE_FOLDER = "train2017"

# CPU
device = "cpu"

print("Chargement du modèle CLIP...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Charger les métadonnées
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Sélectionner une image et une caption
sample = data[0]
image_path = f"{IMAGE_FOLDER}/{sample['file_name']}"
caption = sample["captions"][0]

print("\nImage :", sample["file_name"])
print("Caption :", caption)

# Charger l'image
image = Image.open(image_path).convert("RGB")

# Générer l'embedding de l'image
image_inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_output = model.get_image_features(**image_inputs)
    image_embedding = image_output.pooler_output
# Générer l'embedding du texte
text_inputs = processor(
    text=[caption],
    return_tensors="pt",
    padding=True,
    truncation=True
)

with torch.no_grad():
   text_output = model.get_text_features(**text_inputs)
   text_embedding = text_output.pooler_output

# Normaliser les embeddings
image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=-1)
text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)

# Vérifier les dimensions
print("\n===== RÉSULTATS DU TEST =====")
print("Dimension image :", tuple(image_embedding.shape))
print("Dimension texte :", tuple(text_embedding.shape))
print("Image embedding valide :", torch.isfinite(image_embedding).all().item())
print("Text embedding valide :", torch.isfinite(text_embedding).all().item())

print("\nTest des embeddings terminé !")
