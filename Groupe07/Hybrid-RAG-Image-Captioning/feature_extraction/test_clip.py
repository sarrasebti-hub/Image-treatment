
import torch
import open_clip
from PIL import Image
from pathlib import Path
import json

# Use CPU
device = "cpu"

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

# Select 10 training images
images_dir = Path("dataset/images_train")
image_paths = sorted(images_dir.glob("*.jpg"))[:10]

if not image_paths:
    raise FileNotFoundError("No training images found.")

embeddings = []

with torch.no_grad():
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

        embeddings.append({
            "image": image_path.name,
            "embedding_dimension": features.shape[1],
            "embedding": features.squeeze(0).tolist()
        })

# Save test embeddings
output_path = Path("feature_extraction/test_embeddings.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(embeddings, f)

print("CLIP test completed successfully!")
print("Images processed:", len(embeddings))
print("Embedding dimension:", embeddings[0]["embedding_dimension"])
print("Output:", output_path)
