
import json
import numpy as np
from pathlib import Path

# 1. Chemins
BASE_DIR = Path(__file__).resolve().parent

METADATA_PATH = BASE_DIR / "metadata" / "knowledge_base_metadata.json"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

FILES = [
    "image_embeddings.npy",
    "text_embeddings.npy",
    "image_ids.json",
    "caption_records.json"
]

print("========== CONTROLE QUALITE FINAL ==========")

# 2. Verification des fichiers
for filename in FILES:
    path = EMBEDDINGS_DIR / filename

    assert path.is_file(), f"Fichier manquant : {filename}"
    assert path.stat().st_size > 0, f"Fichier vide : {filename}"

    print("Fichier present :", filename)

# 3. Rechargement des donnees
image_embeddings = np.load(
    EMBEDDINGS_DIR / "image_embeddings.npy",
    allow_pickle=False
)

text_embeddings = np.load(
    EMBEDDINGS_DIR / "text_embeddings.npy",
    allow_pickle=False
)

with open(EMBEDDINGS_DIR / "image_ids.json", encoding="utf-8") as f:
    image_ids = json.load(f)

with open(EMBEDDINGS_DIR / "caption_records.json", encoding="utf-8") as f:
    caption_records = json.load(f)

with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

print("\nRechargement des fichiers : OK")

# 4. Dimensions et types
assert image_embeddings.shape == (10000, 512)
assert text_embeddings.shape == (50000, 512)

assert image_embeddings.dtype == np.float32
assert text_embeddings.dtype == np.float32

print("Dimensions images :", image_embeddings.shape)
print("Dimensions textes :", text_embeddings.shape)
print("Types numeriques : float32")

# 5. Identifiants
expected_ids = [item["image_id"] for item in metadata]

assert len(image_ids) == 10000
assert image_ids == expected_ids
assert len(set(image_ids)) == 10000

print("Identifiants images : OK")

# 6. Correspondance des captions
expected_records = []

for item in metadata:
    for caption in item["captions"]:
        expected_records.append({
            "image_id": item["image_id"],
            "caption": caption
        })

assert len(caption_records) == 50000
assert caption_records == expected_records

print("Correspondance captions : OK")

# 7. Verification numerique
assert np.isfinite(image_embeddings).all()
assert np.isfinite(text_embeddings).all()

image_norms = np.linalg.norm(image_embeddings, axis=1)
text_norms = np.linalg.norm(text_embeddings, axis=1)

assert np.allclose(image_norms, 1.0, atol=1e-4)
assert np.allclose(text_norms, 1.0, atol=1e-4)

print("Valeurs numeriques : OK")
print("Normalisation L2 : OK")

# 8. Coherence
assert len(image_ids) == image_embeddings.shape[0]
assert len(caption_records) == text_embeddings.shape[0]

print("Coherence images / embeddings : OK")
print("Coherence captions / embeddings : OK")

# 9. Rapport final
print("\n========== RAPPORT FINAL ==========")
print("Images :", len(image_ids))
print("Captions :", len(caption_records))
print("Dimension CLIP :", image_embeddings.shape[1])
print("Fichiers sauvegardes :", len(FILES))

print("\nSUCCES : CONTROLE QUALITE VALIDE.")
