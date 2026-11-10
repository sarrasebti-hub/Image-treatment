import json
import time
import faiss
import numpy as np

from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = BASE_DIR / "dataset" / "embeddings" / "image_embeddings.npy"
IDS_PATH = BASE_DIR / "dataset" / "embeddings" / "image_ids.json"

OUTPUT_PATH = BASE_DIR / "embeddings" / "train2017_benchmark.faiss"

print("========== FAISS BUILD BENCHMARK ==========")

# ==========================================
# 2. CHARGEMENT DES EMBEDDINGS
# ==========================================

print("\n[1/5] Chargement des embeddings...")

embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)

with open(IDS_PATH, encoding="utf-8") as f:
    image_ids = json.load(f)

assert embeddings.shape == (10000, 512)
assert embeddings.dtype == np.float32
assert len(image_ids) == 10000
assert np.isfinite(embeddings).all()

print("Shape:", embeddings.shape)
print("Dtype:", embeddings.dtype)

# ==========================================
# 3. VERIFICATION NORMALISATION
# ==========================================

print("\n[2/5] Vérification des vecteurs...")

norms = np.linalg.norm(embeddings, axis=1)

assert np.allclose(norms, 1.0, atol=1e-4)

print("Normalisation L2 : validée")

# ==========================================
# 4. CONSTRUCTION FAISS
# ==========================================

print("\n[3/5] Construction de l'index...")

start_time = time.perf_counter()

index = faiss.IndexFlatIP(512)
index.add(embeddings)

build_time = time.perf_counter() - start_time

assert index.ntotal == 10000
assert index.d == 512

print("Index type:", type(index).__name__)
print("Vectors:", index.ntotal)
print("Dimension:", index.d)
print("Construction time:", round(build_time, 4), "seconds")

# ==========================================
# 5. SAUVEGARDE SEPAREE
# ==========================================

print("\n[4/5] Sauvegarde de l'index benchmark...")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(OUTPUT_PATH))

size_bytes = OUTPUT_PATH.stat().st_size
size_mb = size_bytes / (1024 * 1024)

print("Fichier:", OUTPUT_PATH)
print("Taille:", size_bytes, "bytes")
print("Taille:", round(size_mb, 2), "MiB")

# ==========================================
# 6. VALIDATION FINALE
# ==========================================

print("\n[5/5] Validation...")

reloaded_index = faiss.read_index(str(OUTPUT_PATH))

assert reloaded_index.ntotal == 10000
assert reloaded_index.d == 512

reconstructed = reloaded_index.reconstruct_n(
    0,
    reloaded_index.ntotal
)

assert np.allclose(reconstructed, embeddings, atol=1e-5)

print("Index sauvegardé et rechargé : OK")
print("Alignement des vecteurs : OK")

# ==========================================
# 7. RAPPORT
# ==========================================

print("\n========== RAPPORT FINAL ==========")
print("Images indexées:", index.ntotal)
print("Dimension:", index.d)
print("Index:", type(index).__name__)
print("Construction time:", round(build_time, 4), "seconds")
print("Index size:", round(size_mb, 2), "MiB")

print("\nSUCCESS: FAISS BENCHMARK COMPLETED.")