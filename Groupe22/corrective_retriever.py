
import json
from pathlib import Path

import faiss
import numpy as np
import torch

from transformers import CLIPModel, CLIPTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

FAISS_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"
VAL_IDS_FILE = EMBEDDINGS_DIR / "val2017_ids.json"
VAL_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "val2017_embeddings.npy"
KB_FILE = DATASET_DIR / "knowledge_base_train.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K_FAISS = 20
TOP_K_FINAL = 5
BATCH_SIZE = 32

QUERY_INDEX = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. VERIFICATION DES FICHIERS
# ============================================================

print("Verification des fichiers...")

required_files = [
    FAISS_FILE,
    TRAIN_IDS_FILE,
    VAL_IDS_FILE,
    VAL_EMBEDDINGS_FILE,
    KB_FILE
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

print("Tous les fichiers sont presents.")


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

print("\nChargement des fichiers...")

index = faiss.read_index(str(FAISS_FILE))

with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
    train_ids = json.load(f)

with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
    val_ids = json.load(f)

val_embeddings = np.load(VAL_EMBEDDINGS_FILE)

with open(KB_FILE, "r", encoding="utf-8") as f:
    knowledge_base_list = json.load(f)


# ============================================================
# 4. NORMALISATION DES IDENTIFIANTS
# ============================================================

def normalize_image_id(image_id):
    """
    Convertit un identifiant COCO en chaine de 12 chiffres.

    Exemple :
    34             -> 000000000034
    '34'           -> 000000000034
    '000000000034' -> 000000000034
    """

    return f"{int(image_id):012d}"


knowledge_base = {}

for item in knowledge_base_list:

    image_id = normalize_image_id(item["image_id"])

    knowledge_base[image_id] = item


train_ids = [
    normalize_image_id(image_id)
    for image_id in train_ids
]

val_ids = [
    normalize_image_id(image_id)
    for image_id in val_ids
]


print("Index FAISS :", index.ntotal)
print("Images train :", len(train_ids))
print("Images validation :", len(val_ids))
print("Base de connaissances :", len(knowledge_base))


# ============================================================
# 5. VERIFICATION DE LA COHERENCE
# ============================================================

if index.ntotal != len(train_ids):
    raise ValueError(
        "Le nombre de vecteurs FAISS ne correspond pas "
        "au nombre d'identifiants train."
    )

if len(val_ids) != len(val_embeddings):
    raise ValueError(
        "Le nombre d'identifiants validation ne correspond "
        "pas au nombre de vecteurs validation."
    )

missing_ids = [
    image_id
    for image_id in train_ids
    if image_id not in knowledge_base
]

print("Images train absentes de la base :", len(missing_ids))

if missing_ids:
    print("Exemples d'identifiants absents :", missing_ids[:5])


# ============================================================
# 6. CHARGEMENT DU MODELE CLIP
# ============================================================

print("\nChargement du modele CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)

tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("Modele charge sur :", DEVICE)


# ============================================================
# 7. ENCODAGE DES LEGENDES
# ============================================================

def encode_captions(captions):

    if not captions:
        return np.empty((0, 512), dtype=np.float32)

    all_embeddings = []

    with torch.no_grad():

        for start in range(0, len(captions), BATCH_SIZE):

            batch = captions[start:start + BATCH_SIZE]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )

            tokens = {
                key: value.to(DEVICE)
                for key, value in tokens.items()
            }

            # Encodeur textuel CLIP
            text_outputs = model.text_model(**tokens)

            # Projection dans l'espace commun CLIP
            text_features = model.text_projection(
                text_outputs.pooler_output
            )

            # Normalisation L2
            text_features = torch.nn.functional.normalize(
                text_features,
                p=2,
                dim=1
            )

            all_embeddings.append(
                text_features.cpu().numpy()
            )

    return np.concatenate(
        all_embeddings,
        axis=0
    ).astype(np.float32)


# ============================================================
# 8. RECUPERATION INITIALE FAISS
# ============================================================

def retrieve_candidates(query_embedding, top_k=TOP_K_FAISS):

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    # Vérification et normalisation du vecteur
    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        raise ValueError("Le vecteur de requete est nul.")

    query_embedding = query_embedding / query_norm

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    candidates = []

    validation_ids = set(val_ids)

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        image_id = train_ids[int(idx)]

        # Exclusion des images de validation
        if image_id in validation_ids:
            continue

        entry = knowledge_base.get(image_id)

        if entry is None:
            continue

        captions = entry.get("captions", [])

        if not captions:
            continue

        candidates.append({
            "image_id": image_id,
            "faiss_score": float(score),
            "captions": captions
        })

    return candidates


# ============================================================
# 9. CORRECTIVE RETRIEVAL AVEC CLIP
# ============================================================

def corrective_retrieve(query_embedding):

    print("\nRecherche initiale avec FAISS...")

    candidates = retrieve_candidates(
        query_embedding,
        TOP_K_FAISS
    )

    print(
        "Nombre de candidats recuperes :",
        len(candidates)
    )

    if not candidates:
        print("Aucun candidat trouve.")

        return []

    # Collecte des légendes candidates
    all_captions = []

    for candidate in candidates:
        all_captions.extend(candidate["captions"])

    print(
        "Nombre de legendes a evaluer :",
        len(all_captions)
    )

    # Encodage des légendes
    text_embeddings = encode_captions(all_captions)

    # Normalisation de la requête
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    query_embedding = query_embedding / np.linalg.norm(
        query_embedding
    )

    # Similarité cosinus image-texte
    similarities = (
        text_embeddings @ query_embedding.T
    ).flatten()

    # Association des scores aux légendes
    position = 0

    for candidate in candidates:

        number = len(candidate["captions"])

        candidate_scores = similarities[
            position:position + number
        ]

        best_index = int(
            np.argmax(candidate_scores)
        )

        candidate["best_caption"] = (
            candidate["captions"][best_index]
        )

        candidate["clip_score"] = float(
            candidate_scores[best_index]
        )

        position += number

    # Reclassement des candidats
    candidates.sort(
        key=lambda item: item["clip_score"],
        reverse=True
    )

    return candidates[:TOP_K_FINAL]


# ============================================================
# 10. EXECUTION PRINCIPALE
# ============================================================

def main():

    print("\n========================================")
    print("             CORRECTIVE RAG")
    print("========================================")

    query_id = val_ids[QUERY_INDEX]

    query_embedding = val_embeddings[QUERY_INDEX]

    print("\nImage de validation :", query_id)

    results = corrective_retrieve(query_embedding)

    print("\n========================================")
    print("              RESULTATS")
    print("========================================")

    if not results:
        print("Aucun resultat disponible.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\nRang {rank}")

        print("Image ID :", result["image_id"])

        print(
            "Score FAISS :",
            round(result["faiss_score"], 4)
        )

        print(
            "Score CLIP :",
            round(result["clip_score"], 4)
        )

        print("Legende :", result["best_caption"])

    print("\n========================================")
    print("          EXECUTION TERMINEE")
    print("========================================")


# ============================================================
# 11. POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    main()
