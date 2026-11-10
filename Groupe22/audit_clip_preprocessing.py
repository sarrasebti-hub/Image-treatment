from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPImageProcessor

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "dataset" / "preprocessed" / "val2017"

# Charger une image prétraitée
image_files = sorted(IMAGE_DIR.glob("*.jpg"))

if not image_files:
    raise FileNotFoundError("Aucune image JPG trouvée dans le dossier.")

image_path = image_files[0]

print("=" * 60)
print("AUDIT DU PRETRAITEMENT CLIP")
print("=" * 60)

print(f"\nImage testée : {image_path.name}")

# Ouvrir l'image en RGB
with Image.open(image_path) as img:
    image = img.convert("RGB")

print(f"Mode couleur avant CLIP : {image.mode}")
print(f"Dimensions avant CLIP : {image.size}")

# Charger le processeur CLIP
processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

print("\nConfiguration du processeur :")
print("Resize :", processor.do_resize)
print("Center crop :", processor.do_center_crop)
print("Rescale :", processor.do_rescale)
print("Normalize :", processor.do_normalize)
print("Taille demandée :", processor.size)
print("Taille du crop :", processor.crop_size)
print("Moyenne :", processor.image_mean)
print("Écart-type :", processor.image_std)

# Appliquer le prétraitement
inputs = processor(
    images=image,
    return_tensors="pt"
)

pixel_values = inputs["pixel_values"]

print("\nRésultat du prétraitement :")
print("Shape :", tuple(pixel_values.shape))
print("Type :", pixel_values.dtype)
print("Minimum :", pixel_values.min().item())
print("Maximum :", pixel_values.max().item())

# Vérifier les valeurs
print("\nVérifications :")

print(
    "Tenseur de forme (1, 3, 224, 224) :",
    tuple(pixel_values.shape) == (1, 3, 224, 224)
)

print(
    "Valeurs finies (pas de NaN ou Inf) :",
    torch.isfinite(pixel_values).all().item()
)

print(
    "Valeurs différentes de zéro :",
    torch.count_nonzero(pixel_values).item() > 0
)

print(
    "Normalisation activée :",
    processor.do_normalize
)

print("\nAUDIT TERMINE")