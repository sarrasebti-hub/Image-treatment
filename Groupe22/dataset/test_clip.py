
import torch
from transformers import CLIPModel, CLIPProcessor

# 1. Choisir le modèle CLIP
MODEL_NAME = "openai/clip-vit-base-patch32"

print("Chargement du modèle CLIP...")
print("Cela peut prendre quelques minutes au premier lancement.")

# 2. Utiliser le CPU
device = "cpu"

# 3. Charger le modèle et son processeur
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

# 4. Vérifier le chargement
print("\n===== TEST CLIP =====")
print("Modèle :", MODEL_NAME)
print("Device :", device)
print("Dimension des embeddings :", model.config.projection_dim)
print("Chargement réussi !")
