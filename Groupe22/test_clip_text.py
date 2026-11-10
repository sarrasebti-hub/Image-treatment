
import torch
from transformers import CLIPModel, CLIPTokenizer

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Chargement du modèle CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

captions = [
    "A living room with a television and chairs.",
    "A dog running in a park.",
    "A kitchen with a dining table."
]

inputs = tokenizer(
    captions,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}

with torch.inference_mode():
    text_features = model.get_text_features(**inputs)

    # Compatible avec les différentes versions de Transformers
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output

    text_features = torch.nn.functional.normalize(
        text_features,
        p=2,
        dim=1
    )

print("\nTest réussi !")
print("Shape :", text_features.shape)
print("NaN :", torch.isnan(text_features).any().item())
print("Device :", DEVICE)
