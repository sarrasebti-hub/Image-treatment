
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

INDEX_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"
VAL_IDS_FILE = EMBEDDINGS_DIR / "val2017_ids.json"
VAL_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "val2017_embeddings.npy"
CALIBRATION_FILE = EMBEDDINGS_DIR / "crag_calibration_results.json"
KB_FILE = DATASET_DIR / "knowledge_base_train.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

LOWER_THRESHOLD = 0.3170
UPPER_THRESHOLD = 0.3402

INITIAL_K = 20
EXPANDED_K = 50

TOP_CAPTIONS = 5
NUMBER_OF_QUERIES = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. CLASSIFICATION
# ==========================================

def classify_relevance(score):

    if score >= UPPER_THRESHOLD:
        return "CORRECT"

    elif score >= LOWER_THRESHOLD:
        return "AMBIGUOUS"

    return "INCORRECT"


# ==========================================
# 3. NORMALISATION DES IDENTIFIANTS
# ==========================================

def normalize_image_id(image_id):
    return f"{int(image_id):012d}"


# ==========================================
# 4. CHARGEMENT DES DONNEES
# ==========================================

def load_data():

    print("\nVerification des fichiers...")

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

    print("Chargement des fichiers...")

    index = faiss.read_index(str(INDEX_FILE))

    with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
        train_ids = json.load(f)

    with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
        val_ids = json.load(f)

    val_embeddings = np.load(VAL_EMBEDDINGS_FILE)

    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        calibration_results = json.load(f)

    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb_list = json.load(f)

    train_ids = [
        normalize_image_id(x) for x in train_ids
    ]

    val_ids = [
        normalize_image_id(x) for x in val_ids
    ]

    knowledge_base = {
        normalize_image_id(item["image_id"]): item
        for item in kb_list
    }

    if index.ntotal != len(train_ids):
        raise ValueError("Index FAISS et IDs train incompatibles.")

    if len(val_ids) != len(val_embeddings):
        raise ValueError("IDs validation et embeddings incompatibles.")

    print("Index FAISS :", index.ntotal)
    print("Images validation :", len(val_ids))
    print("Base de connaissances :", len(knowledge_base))

    return (
        index,
        train_ids,
        val_ids,
        val_embeddings,
        calibration_results,
        knowledge_base
    )


# ==========================================
# 5. CHARGEMENT DU MODELE CLIP
# ==========================================

def load_clip():

    print("\nChargement du modele CLIP...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    print("Modele charge sur :", DEVICE)

    return model, tokenizer


# ==========================================
# 6. ENCODAGE DES LEGENDES
# ==========================================

def encode_captions(captions, model, tokenizer):

    tokens = tokenizer(
        captions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    tokens = {
        key: value.to(DEVICE)
        for key, value in tokens.items()
    }

    with torch.no_grad():

        outputs = model.text_model(**tokens)

        features = model.text_projection(
            outputs.pooler_output
        )

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1
        )

    return features.cpu().numpy()


# ==========================================
# 7. CLASSIFICATION DES CANDIDATS
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

    scores, indices = index.search(query, top_k)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        image_id = train_ids[idx]

        # Exclusion des images de validation
        if image_id in val_ids:
            continue

        if image_id not in knowledge_base:
            continue

        captions = knowledge_base[image_id]["captions"]

        candidates.append({
            "image_id": image_id,
            "faiss_score": float(score),
            "captions": captions
        })

    return candidates


# ==========================================
# 8. RE-EVALUATION ET RE-RANKING CLIP
# ==========================================

def rerank_captions(
    candidates,
    query_embedding,
    model,
    tokenizer
):

    all_captions = []
    metadata = []

    for candidate in candidates:

        for caption in candidate["captions"]:

            all_captions.append(caption)

            metadata.append({
                "image_id": candidate["image_id"],
                "faiss_score": candidate["faiss_score"],
                "caption": caption
            })

    if not all_captions:
        return []

    text_embeddings = encode_captions(
        all_captions,
        model,
        tokenizer
    )

    query = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    query = query / (
        np.linalg.norm(query) + 1e-12
    )

    clip_scores = np.dot(
        text_embeddings,
        query.T
    ).flatten()

    results = []

    for item, score in zip(metadata, clip_scores):

        results.append({
            "image_id": item["image_id"],
            "faiss_score": item["faiss_score"],
            "clip_score": float(score),
            "caption": item["caption"]
        })

    results.sort(
        key=lambda x: x["clip_score"],
        reverse=True
    )

    return results


# ==========================================
# 9. PIPELINE CRAG
# ==========================================

def process_query(
    query_id,
    query_embedding,
    initial_score,
    index,
    train_ids,
    val_ids,
    knowledge_base,
    model,
    tokenizer
):

    category = classify_relevance(initial_score)

    print("\n----------------------------------------")
    print("Image :", query_id)
    print(f"Score initial : {initial_score:.4f}")
    print("Classification :", category)

    if category == "CORRECT":

        top_k = INITIAL_K
        action = "KEEP_INITIAL_RESULTS"

    elif category == "AMBIGUOUS":

        top_k = EXPANDED_K
        action = "EXPANDED_RETRIEVAL"

    else:

        top_k = EXPANDED_K
        action = "FALLBACK_EXPANDED_RETRIEVAL"

    print("Action :", action)
    print("Nombre de candidats demande :", top_k)

    candidates = retrieve_candidates(
        index,
        query_embedding,
        train_ids,
        val_ids,
        knowledge_base,
        top_k
    )

    print("Candidats recuperes :", len(candidates))

    ranked_results = rerank_captions(
        candidates,
        query_embedding,
        model,
        tokenizer
    )

    print("Nombre de legendes re-evaluees :", len(ranked_results))

    print("\nTOP 5 DES MEILLEURES LEGENDES")

    for rank, result in enumerate(
        ranked_results[:TOP_CAPTIONS],
        start=1
    ):

        print(f"\nRang {rank}")
        print("Image ID :", result["image_id"])
        print(f"Score FAISS : {result['faiss_score']:.4f}")
        print(f"Score CLIP : {result['clip_score']:.4f}")
        print("Legende :", result["caption"])

    return {
        "query_id": query_id,
        "initial_score": initial_score,
        "classification": category,
        "action": action,
        "number_of_candidates": len(candidates),
        "number_of_captions": len(ranked_results),
        "top_results": ranked_results[:TOP_CAPTIONS]
    }


# ==========================================
# 10. EXECUTION PRINCIPALE
# ==========================================

def main():

    print("\n========================================")
    print("          CRAG RE-RANKING")
    print("========================================")

    (
        index,
        train_ids,
        val_ids,
        val_embeddings,
        calibration_results,
        knowledge_base
    ) = load_data()

    model, tokenizer = load_clip()

    val_id_to_index = {
        image_id: i
        for i, image_id in enumerate(val_ids)
    }

    final_results = []

    for item in calibration_results[:NUMBER_OF_QUERIES]:

        query_id = normalize_image_id(item["query_id"])

        if query_id not in val_id_to_index:
            print("Image validation introuvable :", query_id)
            continue

        query_index = val_id_to_index[query_id]

        query_embedding = val_embeddings[query_index]

        result = process_query(
            query_id,
            query_embedding,
            float(item["best_clip_score"]),
            index,
            train_ids,
            val_ids,
            knowledge_base,
            model,
            tokenizer
        )

        final_results.append(result)

    output_file = EMBEDDINGS_DIR / "crag_reranking_results.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            final_results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n========================================")
    print("RE-RANKING TERMINE")
    print("Nombre d'images traitees :", len(final_results))
    print("Resultats sauvegardes dans :")
    print(output_file)
    print("========================================")


if __name__ == "__main__":
    main()
