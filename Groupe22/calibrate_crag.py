
import json
from pathlib import Path

import faiss
import numpy as np
import torch

from transformers import CLIPModel, CLIPTokenizer


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

FAISS_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"
VAL_IDS_FILE = EMBEDDINGS_DIR / "val2017_ids.json"
VAL_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "val2017_embeddings.npy"
KB_FILE = DATASET_DIR / "knowledge_base_train.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K = 20
BATCH_SIZE = 32

# Nombre d'images de validation a analyser
N_VALIDATION = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# 2. CHARGEMENT DES DONNEES
# =====================================================

print("Chargement des donnees...")

index = faiss.read_index(str(FAISS_FILE))

with open(TRAIN_IDS_FILE, encoding="utf-8") as f:
    train_ids = json.load(f)

with open(VAL_IDS_FILE, encoding="utf-8") as f:
    val_ids = json.load(f)

with open(KB_FILE, encoding="utf-8") as f:
    kb_list = json.load(f)

val_embeddings = np.load(VAL_EMBEDDINGS_FILE)


def normalize_id(image_id):
    return f"{int(image_id):012d}"


train_ids = [normalize_id(x) for x in train_ids]
val_ids = [normalize_id(x) for x in val_ids]

knowledge_base = {
    normalize_id(item["image_id"]): item
    for item in kb_list
}

if index.ntotal != len(train_ids):
    raise ValueError("Incoherence entre FAISS et train IDs.")

if len(val_ids) != len(val_embeddings):
    raise ValueError("Incoherence entre val IDs et embeddings.")

n = min(N_VALIDATION, len(val_ids))

print("Images de validation analysees :", n)


# =====================================================
# 3. CHARGEMENT DE CLIP
# =====================================================

print("\nChargement de CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("CLIP charge sur :", DEVICE)


# =====================================================
# 4. ENCODAGE DES LEGENDES
# =====================================================

def encode_captions(captions):

    embeddings = []

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

            outputs = model.text_model(**tokens)

            features = model.text_projection(
                outputs.pooler_output
            )

            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=1
            )

            embeddings.append(features.cpu().numpy())

    if not embeddings:
        return np.empty((0, 512), dtype=np.float32)

    return np.concatenate(embeddings).astype(np.float32)


# =====================================================
# 5. ANALYSE DES IMAGES DE VALIDATION
# =====================================================

all_best_scores = []
results = []

validation_set = set(val_ids)

print("\nDebut de la calibration...\n")

for query_index in range(n):

    query_id = val_ids[query_index]

    query = np.asarray(
        val_embeddings[query_index],
        dtype=np.float32
    ).reshape(1, -1)

    query = query / np.linalg.norm(query)

    scores, indices = index.search(query, TOP_K)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        image_id = train_ids[int(idx)]

        if image_id in validation_set:
            continue

        entry = knowledge_base.get(image_id)

        if entry is None:
            continue

        captions = entry.get("captions", [])

        if not captions:
            continue

        candidates.append({
            "image_id": image_id,
            "captions": captions
        })

    if not candidates:
        print(f"{query_id}: aucun candidat")
        continue

    all_captions = []

    for candidate in candidates:
        all_captions.extend(candidate["captions"])

    text_embeddings = encode_captions(all_captions)

    similarities = (
        text_embeddings @ query.T
    ).flatten()

    position = 0
    query_scores = []

    for candidate in candidates:

        count = len(candidate["captions"])

        candidate_scores = similarities[
            position:position + count
        ]

        best_score = float(np.max(candidate_scores))

        query_scores.append(best_score)

        position += count

    best_query_score = max(query_scores)

    all_best_scores.append(best_query_score)

    results.append({
        "query_id": query_id,
        "best_clip_score": best_query_score,
        "candidate_count": len(candidates)
    })

    print(
        f"{query_index + 1}/{n} | "
        f"Image: {query_id} | "
        f"Meilleur score: {best_query_score:.4f}"
    )


# =====================================================
# 6. STATISTIQUES DE CALIBRATION
# =====================================================

if not all_best_scores:
    raise RuntimeError("Aucun score obtenu.")

scores_array = np.array(all_best_scores)

print("\n========================================")
print("       STATISTIQUES DE CALIBRATION")
print("========================================")

print("Nombre d'images analysees :", len(scores_array))

print("Score minimum :", round(float(np.min(scores_array)), 4))
print("Score maximum :", round(float(np.max(scores_array)), 4))
print("Score moyen :", round(float(np.mean(scores_array)), 4))
print("Mediane :", round(float(np.median(scores_array)), 4))

print("Percentile 10 :", round(float(np.percentile(scores_array, 10)), 4))
print("Percentile 25 :", round(float(np.percentile(scores_array, 25)), 4))
print("Percentile 75 :", round(float(np.percentile(scores_array, 75)), 4))
print("Percentile 90 :", round(float(np.percentile(scores_array, 90)), 4))


# =====================================================
# 7. SAUVEGARDE DES RESULTATS
# =====================================================

output_file = EMBEDDINGS_DIR / "crag_calibration_results.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nResultats sauvegardes dans :")
print(output_file)

print("\nCalibration terminee.")
