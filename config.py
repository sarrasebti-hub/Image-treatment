"""Configuration centrale du projet. Modifie les valeurs ICI (pas dans les scripts)."""
import os
from pathlib import Path

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:          # permet de lancer les étapes 0-3 sans torch
    DEVICE = "cpu"


def _int(name, default):
    """Permet de surcharger une valeur par variable d'environnement (utile pour tester vite)."""
    return int(os.getenv(name, default))


# ----------------------------- Chemins -----------------------------
ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
RAW = ROOT / "data" / "raw"
ANN_DIR = RAW / "annotations"
TRAIN_IMG_DIR = RAW / "train2017"   # images KB + validation (si le prof les fournit : mets leur dossier ici)
VAL_IMG_DIR = RAW / "val2017"       # images du TEST
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
METADATA = ROOT / "data" / "metadata"
EMB_DIR = ROOT / "embeddings"
VDB_DIR = ROOT / "vector_db"
RESULTS = ROOT / "results"
VIZ = RESULTS / "visualizations"
REPORTS = RESULTS / "reports"

# ------------------ Tailles des splits (A CONFIRMER avec le prof) ------------------
SEED = 42
N_KB = _int("N_KB", 10_000)
N_VAL = _int("N_VAL", 100)           # validation DISJOINTE de la KB (si assez d'images)
N_TEST = _int("N_TEST", 100)         # 100 ou 1000 ? -> à confirmer
N_CAPTIONS = 5                       # captions de référence par image
N_CANDIDATES = _int("N_CANDIDATES", N_KB + N_VAL + 200)        # marge pour les suppressions
N_TEST_CANDIDATES = _int("N_TEST_CANDIDATES", max(300, 3 * N_TEST))

# ----------------------------- Nettoyage -----------------------------
PHASH_THRESHOLD = 2                  # distance de Hamming max => "quasi-doublon visuel"
REMOVE_VISUAL_DUPLICATES = False     # False = on signale seulement (revue manuelle)

# ----------------------------- Modèle -----------------------------
CLIP_MODEL = "openai/clip-vit-base-patch32"   # ~600 Mo, tourne sur CPU
BATCH_SIZE = _int("BATCH_SIZE", 64)

for _d in [ANN_DIR, TRAIN_IMG_DIR, VAL_IMG_DIR, PROCESSED, SPLITS, METADATA,
           EMB_DIR, VDB_DIR, VIZ, REPORTS]:
    _d.mkdir(parents=True, exist_ok=True)
