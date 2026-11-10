
import json
from pathlib import Path

import faiss
import numpy as np


# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_DIR = BASE_DIR / "dataset" / "embeddings"

INDEX_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"
VAL_IDS_FILE = EMBEDDINGS_DIR / "val2017_ids.json"
VAL_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "val2017_embeddings.npy"

CALIBRATION_FILE = EMBEDDINGS_DIR / "crag_calibration_results.json"

KB_FILE = BASE_DIR / "dataset" / "knowledge_base_train.json"


# ==========================================
# 2. PARAMETRES CRAG
# ==========================================

LOWER_THRESHOLD = 0.3170
UPPER_THRESHOLD = 0.3402

INITIAL_K = 20
EXPANDED_K = 50

# Nombre de requetes de validation a tester
NUMBER_OF_QUERIES = 20


# ==========================================
# 3. CLASSIFICATION
# ==========================================

def classify_relevance(score):

    if score >= UPPER_THRESHOLD:
        return "CORRECT"

    elif score >= LOWER_THRESHOLD:
        return "AMBIGUOUS"

    else:
        return "INCORRECT"


# ==========================================
# 4. NORMALISATION DES IDENTIFIANTS
# ==========================================

def normalize_image_id(image_id):
    return f"{int(image_id):012d}"


# ==========================================
# 5. CHARGEMENT DES DONNEES
# ==========================================

def load_data():

    print("\nChargement des fichiers...")

    required_files = [
        INDEX_FILE,
        TRAIN_IDS_FILE,
        VAL_IDS_FILE,
        VAL_EMBEDDINGS_FILE,
        CALIBRATION_FILE,
        KB_FILE
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

    index = faiss.read_index(str(INDEX_FILE))

    with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
        train_ids = json.load(f)

    with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
        val_ids = json.load(f)

    val_embeddings = np.load(VAL_EMBEDDINGS_FILE)

    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        calibration_results = json.load(f)

    with open(KB_FILE, "r", encoding="utf-8") as f:
        knowledge_base_list = json.load(f)

    train_ids = [
        normalize_image_id(x) for x in train_ids
    ]

    val_ids = [
        normalize_image_id(x) for x in val_ids
    ]

    knowledge_base = {
        normalize_image_id(item["image_id"]): item
        for item in knowledge_base_list
    }

    print("Index FAISS :", index.ntotal)
    print("Images train :", len(train_ids))
    print("Images validation :", len(val_ids))
    print("Base de connaissances :", len(knowledge_base))

    if index.ntotal != len(train_ids):
        raise ValueError(
            "La taille de l'index FAISS ne correspond pas "
            "au nombre d'identifiants train."
        )

    if len(val_ids) != len(val_embeddings):
        raise ValueError(
            "Le nombre d'identifiants validation ne correspond "
            "pas au nombre d'embeddings."
        )

    return (
        index,
        train_ids,
        val_ids,
        val_embeddings,
        calibration_results,
        knowledge_base
    )


# ==========================================
# 6. RECUPERATION FAISS
# ==========================================

def retrieve_candidates(
    index,
    query_embedding,
    train_ids,
    val_ids,
    knowledge_base,
    top_k
):

    query = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    # Recherche dans FAISS
    scores, indices = index.search(query, top_k)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        image_id = train_ids[idx]

        # Protection contre la fuite des images validation
        if image_id in val_ids:
            continue

        if image_id not in knowledge_base:
            continue

        candidates.append({
            "image_id": image_id,
            "faiss_score": float(score),
            "captions": knowledge_base[image_id]["captions"]
        })

    return candidates


# ==========================================
# 7. ACTIONS CORRECTIVES
# ==========================================

def apply_correction(
    category,
    index,
    query_embedding,
    train_ids,
    val_ids,
    knowledge_base
):

    print("\nAction corrective :", category)

    if category == "CORRECT":

        print("Action : conservation de la recherche initiale.")

        candidates = retrieve_candidates(
            index,
            query_embedding,
            train_ids,
            val_ids,
            knowledge_base,
            INITIAL_K
        )

        action = "KEEP_INITIAL_RESULTS"

    elif category == "AMBIGUOUS":

        print(
            "Action : elargissement de la recherche "
            "de Top 20 a Top 50."
        )

        candidates = retrieve_candidates(
            index,
            query_embedding,
            train_ids,
            val_ids,
            knowledge_base,
            EXPANDED_K
        )

        action = "EXPANDED_RETRIEVAL"

    else:

        print(
            "Action : recherche alternative de secours "
            "avec un classement FAISS elargi."
        )

        # Strategie de secours provisoire :
        # on explore un ensemble plus large de candidats.
        # Ce n'est pas encore une strategie independante
        # de l'encodeur CLIP.
        candidates = retrieve_candidates(
            index,
            query_embedding,
            train_ids,
            val_ids,
            knowledge_base,
            EXPANDED_K
        )

        action = "FALLBACK_EXPANDED_RETRIEVAL"

    return candidates, action


# ==========================================
# 8. EXECUTION DU CRAG
# ==========================================

def main():

    print("\n========================================")
    print("       CRAG CORRECTIVE RETRIEVAL")
    print("========================================")

    (
        index,
        train_ids,
        val_ids,
        val_embeddings,
        calibration_results,
        knowledge_base
    ) = load_data()

    val_id_to_index = {
        image_id: i
        for i, image_id in enumerate(val_ids)
    }

    summary = {
        "CORRECT": 0,
        "AMBIGUOUS": 0,
        "INCORRECT": 0
    }

    for item in calibration_results[:NUMBER_OF_QUERIES]:

        query_id = normalize_image_id(item["query_id"])
        score = float(item["best_clip_score"])

        category = classify_relevance(score)

        summary[category] += 1

        print("\n----------------------------------------")
        print("Image :", query_id)
        print(f"Score CLIP : {score:.4f}")
        print("Classification :", category)

        if query_id not in val_id_to_index:
            print("Image absente des embeddings validation.")
            continue

        query_index = val_id_to_index[query_id]

        query_embedding = val_embeddings[query_index]

        candidates, action = apply_correction(
            category,
            index,
            query_embedding,
            train_ids,
            val_ids,
            knowledge_base
        )

        print("Action appliquee :", action)
        print("Nombre de candidats :", len(candidates))

        for rank, candidate in enumerate(candidates[:5], start=1):

            print(
                f"{rank}. Image {candidate['image_id']} "
                f"| FAISS : {candidate['faiss_score']:.4f}"
            )

            if candidate["captions"]:
                print(
                    "   Legende :",
                    candidate["captions"][0]
                )

    print("\n========================================")
    print("       BILAN DES CLASSIFICATIONS")
    print("========================================")

    total = sum(summary.values())

    for category, count in summary.items():

        percentage = (
            count / total * 100
            if total > 0 else 0
        )

        print(
            f"{category} : {count}/{total} "
            f"({percentage:.1f}%)"
        )

    print("\nCorrection CRAG terminee.")


if __name__ == "__main__":
    main()
