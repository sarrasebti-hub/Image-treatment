
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

FAISS_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"
VAL_IDS_FILE = EMBEDDINGS_DIR / "val2017_ids.json"
VAL_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "val2017_embeddings.npy"

CALIBRATION_FILE = EMBEDDINGS_DIR / "crag_calibration_results.json"
KB_FILE = DATASET_DIR / "knowledge_base_train.json"

OUTPUT_FILE = EMBEDDINGS_DIR / "crag_diversification_results.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

# Seuils provisoires de calibration
THRESHOLD_CORRECT = 0.3402
THRESHOLD_AMBIGUOUS = 0.3170

# Nombre de candidats FAISS
TOP_K_CORRECT = 20
TOP_K_OTHER = 50

# Nombre de légendes finales
FINAL_K = 5

# Paramètre MMR
LAMBDA_MMR = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def normalize_image_id(image_id):
    return f"{int(image_id):012d}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_retrieval(score):

    if score >= THRESHOLD_CORRECT:
        return "CORRECT"

    elif score >= THRESHOLD_AMBIGUOUS:
        return "AMBIGUOUS"

    return "INCORRECT"


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

print("=" * 60)
print("CRAG - DIVERSIFICATION AVEC MMR")
print("=" * 60)

print("\n[1/7] Chargement des fichiers...")

index = faiss.read_index(str(FAISS_FILE))

train_ids = [
    normalize_image_id(x)
    for x in load_json(TRAIN_IDS_FILE)
]

val_ids = [
    normalize_image_id(x)
    for x in load_json(VAL_IDS_FILE)
]

val_embeddings = np.load(
    VAL_EMBEDDINGS_FILE
).astype("float32")

calibration_results = load_json(CALIBRATION_FILE)

knowledge_base_list = load_json(KB_FILE)

knowledge_base = {
    normalize_image_id(item["image_id"]): item
    for item in knowledge_base_list
}

print("Index FAISS :", index.ntotal)
print("Images train :", len(train_ids))
print("Images validation :", len(val_ids))
print("Entrées knowledge base :", len(knowledge_base))


# ============================================================
# 4. VÉRIFICATIONS DES DONNÉES
# ============================================================

print("\n[2/7] Vérification des données...")

if index.ntotal != len(train_ids):
    raise ValueError(
        "La taille de l'index FAISS ne correspond pas aux IDs train."
    )

if len(val_ids) != len(val_embeddings):
    raise ValueError(
        "Les IDs validation et les embeddings ne correspondent pas."
    )

missing_ids = [
    image_id
    for image_id in train_ids
    if image_id not in knowledge_base
]

if missing_ids:
    raise ValueError(
        f"{len(missing_ids)} images train sont absentes de la knowledge base."
    )

validation_leakage = (
    set(val_ids).intersection(set(knowledge_base.keys()))
)

if validation_leakage:
    raise ValueError(
        f"Fuite de données détectée : {len(validation_leakage)} "
        "IDs validation sont présents dans la knowledge base."
    )

print("Vérifications réussies.")


# Vérification du format du fichier de calibration
if not isinstance(calibration_results, list):
    raise ValueError(
        "Le fichier de calibration doit contenir une liste."
    )

if len(calibration_results) == 0:
    raise ValueError(
        "Le fichier de calibration est vide."
    )

for item in calibration_results:

    if "query_id" not in item:
        raise ValueError(
            "Le champ query_id est absent du fichier de calibration."
        )

    if "best_clip_score" not in item:
        raise ValueError(
            "Le champ best_clip_score est absent du fichier de calibration."
        )

print("Format du fichier de calibration vérifié.")


# ============================================================
# 5. CHARGEMENT DU MODÈLE CLIP
# ============================================================

print("\n[3/7] Chargement du modèle CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)

tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("CLIP chargé sur :", DEVICE)


# ============================================================
# 6. ENCODAGE DES LÉGENDES
# ============================================================

def encode_captions(captions, batch_size=64):

    all_embeddings = []

    with torch.no_grad():

        for start in range(0, len(captions), batch_size):

            batch = captions[start:start + batch_size]

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

            text_outputs = model.text_model(**tokens)

            text_features = model.text_projection(
                text_outputs.pooler_output
            )

            text_features = torch.nn.functional.normalize(
                text_features,
                p=2,
                dim=1
            )

            all_embeddings.append(
                text_features.cpu().numpy()
            )

    if not all_embeddings:
        return np.empty((0, 512), dtype="float32")

    return np.vstack(all_embeddings).astype("float32")


# ============================================================
# 7. SÉLECTION MMR
# ============================================================

def mmr_selection(
    captions_data,
    query_embedding,
    caption_embeddings,
    final_k=5,
    lambda_mmr=0.9
):

    # Pertinence image-légende
    relevance_scores = (
        caption_embeddings @ query_embedding
    )

    selected_indices = []

    selected_image_ids = set()

    remaining_indices = set(
        range(len(captions_data))
    )

    while len(selected_indices) < final_k and remaining_indices:

        best_index = None

        best_mmr_score = -float("inf")

        for idx in remaining_indices:

            source_image_id = captions_data[idx]["image_id"]

            # Une seule légende par image source
            if source_image_id in selected_image_ids:
                continue

            relevance = float(
                relevance_scores[idx]
            )

            # Calcul de la redondance
            if selected_indices:

                selected_embeddings = caption_embeddings[
                    selected_indices
                ]

                similarities = (
                    caption_embeddings[idx] @ selected_embeddings.T
                )

                redundancy = float(
                    np.max(similarities)
                )

            else:

                redundancy = 0.0

            # Formule MMR
            mmr_score = (
                lambda_mmr * relevance
                - (1 - lambda_mmr) * redundancy
            )

            if mmr_score > best_mmr_score:

                best_mmr_score = mmr_score
                best_index = idx

        if best_index is None:
            break

        selected_indices.append(best_index)

        selected_image_ids.add(
            captions_data[best_index]["image_id"]
        )

        remaining_indices.remove(best_index)

    selected_results = []

    for idx in selected_indices:

        selected_results.append({

            "image_id": captions_data[idx]["image_id"],

            "caption": captions_data[idx]["caption"],

            "relevance_score": round(
                float(relevance_scores[idx]), 4
            )

        })

    return selected_results


# ============================================================
# 8. PRÉPARATION DES RÉSULTATS DE CALIBRATION
# ============================================================

print("\n[4/7] Préparation des images de validation...")

# IMPORTANT :
# Le fichier de calibration utilise query_id et best_clip_score.

results_by_id = {
    normalize_image_id(item["query_id"]): item
    for item in calibration_results
}

results = []

classification_counts = {
    "CORRECT": 0,
    "AMBIGUOUS": 0,
    "INCORRECT": 0
}

number_to_process = min(
    20,
    len(val_ids)
)

print("Images à traiter :", number_to_process)


# ============================================================
# 9. TRAITEMENT CRAG + MMR
# ============================================================

for query_id in val_ids[:number_to_process]:

    print("\n" + "-" * 60)

    print("Image requête :", query_id)

    if query_id not in results_by_id:

        print("Calibration absente. Image ignorée.")
        continue

    calibration_item = results_by_id[query_id]

    # Champ corrigé : best_clip_score
    initial_score = float(
        calibration_item["best_clip_score"]
    )

    classification = classify_retrieval(
        initial_score
    )

    classification_counts[classification] += 1

    if classification == "CORRECT":

        top_k = TOP_K_CORRECT
        action = "RETRIEVE_TOP_20"

    elif classification == "AMBIGUOUS":

        top_k = TOP_K_OTHER
        action = "EXPANDED_RETRIEVAL_TOP_50"

    else:

        top_k = TOP_K_OTHER
        action = "FALLBACK_RETRIEVAL_TOP_50"

    print("Score initial :", round(initial_score, 4))
    print("Classification :", classification)
    print("Action :", action)

    # Embedding de l'image requête
    query_index = val_ids.index(query_id)

    query_embedding = val_embeddings[
        query_index
    ].reshape(1, -1)

    norm = np.linalg.norm(query_embedding)

    if norm == 0:
        print("Embedding nul. Image ignorée.")
        continue

    query_embedding = (
        query_embedding / norm
    ).astype("float32")[0]

    # Recherche FAISS
    search_scores, search_indices = index.search(
        query_embedding.reshape(1, -1),
        top_k
    )

    candidate_ids = []

    excluded_validation_ids = set(val_ids)

    for idx in search_indices[0]:

        if idx < 0:
            continue

        candidate_id = train_ids[idx]

        if candidate_id in excluded_validation_ids:
            continue

        if candidate_id not in knowledge_base:
            continue

        candidate_ids.append(candidate_id)

    # Récupération des légendes
    captions_data = []

    for candidate_id in candidate_ids:

        item = knowledge_base[candidate_id]

        for caption in item["captions"]:

            captions_data.append({

                "image_id": candidate_id,

                "caption": caption

            })

    if not captions_data:

        print("Aucune légende candidate trouvée.")
        continue

    print("Images candidates :", len(candidate_ids))
    print("Légendes candidates :", len(captions_data))

    # Encodage des légendes avec CLIP
    captions = [
        item["caption"]
        for item in captions_data
    ]

    caption_embeddings = encode_captions(
        captions
    )

    # Diversification MMR
    selected_results = mmr_selection(

        captions_data=captions_data,

        query_embedding=query_embedding,

        caption_embeddings=caption_embeddings,

        final_k=FINAL_K,

        lambda_mmr=LAMBDA_MMR

    )

    print("\nLégendes sélectionnées par MMR :")

    for rank, item in enumerate(
        selected_results,
        start=1
    ):

        print(
            f"{rank}. Image {item['image_id']} "
            f"| Score {item['relevance_score']}"
        )

        print("   ", item["caption"])

    results.append({

        "query_image_id": query_id,

        "initial_score": round(initial_score, 4),

        "classification": classification,

        "action": action,

        "lambda_mmr": LAMBDA_MMR,

        "selected_captions": selected_results

    })


# ============================================================
# 10. SAUVEGARDE DES RÉSULTATS
# ============================================================

print("\n[5/7] Sauvegarde des résultats...")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 11. RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 60)

print("RÉSUMÉ DE LA DIVERSIFICATION MMR")

print("=" * 60)

print("Images traitées :", len(results))

print("\nClassification :")

for label, count in classification_counts.items():

    print(f"  {label} : {count}")

print("\nNombre de légendes sélectionnées :")

if results:

    total_captions = sum(

        len(item["selected_captions"])

        for item in results

    )

    print("Total :", total_captions)

    print(
        "Moyenne par image :",
        round(total_captions / len(results), 2)
    )

print("\nFichier enregistré :")

print(OUTPUT_FILE)

print("\nDIVERSIFICATION MMR TERMINEE")
