
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

MMR_LAMBDAS = [0.5, 0.7, 0.9]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FILE = EMBEDDINGS_DIR / "mmr_parameter_results.json"


# ============================================================
# 2. UTILITAIRES
# ============================================================

def normalize_image_id(image_id):
    return str(image_id).zfill(12)


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Impossible de normaliser un vecteur nul.")

    return vector / norm


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

print("\nLoading dataset...")

index = faiss.read_index(str(FAISS_FILE))

with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as file:
    train_ids = json.load(file)

with open(VAL_IDS_FILE, "r", encoding="utf-8") as file:
    val_ids = json.load(file)

val_embeddings = np.load(VAL_EMBEDDINGS_FILE)

with open(KB_FILE, "r", encoding="utf-8") as file:
    knowledge_base_list = json.load(file)

train_ids = [normalize_image_id(x) for x in train_ids]
val_ids = [normalize_image_id(x) for x in val_ids]

knowledge_base = {
    normalize_image_id(item["image_id"]): item
    for item in knowledge_base_list
}

# Vérification des données

assert index.ntotal == len(train_ids), "Erreur dans les IDs du FAISS."
assert len(val_ids) == len(val_embeddings), "Erreur dans les données Validation."
assert len(knowledge_base) == len(train_ids), "Erreur dans la Knowledge Base."
assert set(train_ids).isdisjoint(set(val_ids)), "Fuite entre Train et Validation."

print("FAISS vectors:", index.ntotal)
print("Training images:", len(train_ids))
print("Validation images:", len(val_ids))
print("Knowledge Base:", len(knowledge_base))


# ============================================================
# 4. CHARGEMENT DU MODELE CLIP
# ============================================================

print("\nLoading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("Model loaded on:", DEVICE)


# ============================================================
# 5. ENCODAGE DES CAPTIONS
# ============================================================

def encode_captions(captions):

    if not captions:
        return np.empty((0, 512), dtype=np.float32)

    all_embeddings = []

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

        with torch.no_grad():

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
            text_features.cpu().numpy().astype(np.float32)
        )

    return np.concatenate(all_embeddings, axis=0)


# ============================================================
# 6. RECUPERATION DES CANDIDATS AVEC FAISS
# ============================================================

def retrieve_candidates(query_embedding, top_k=TOP_K_FAISS):

    query_embedding = normalize_vector(query_embedding)
    query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    validation_set = set(val_ids)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        image_id = train_ids[idx]

        # Exclure les images de validation
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
            "faiss_score": float(score),
            "captions": captions
        })

    return candidates


# ============================================================
# 7. ALGORITHME MMR
# ============================================================

def apply_mmr(query_embedding, candidates, lambda_mmr):

    if not candidates:
        return []

    all_captions = []
    image_ids = []

    for candidate in candidates:

        for caption in candidate["captions"]:

            all_captions.append(caption)
            image_ids.append(candidate["image_id"])

    if not all_captions:
        return []

    text_embeddings = encode_captions(all_captions)

    query_embedding = normalize_vector(query_embedding)

    relevance_scores = (
        text_embeddings @ query_embedding.reshape(-1, 1)
    ).flatten()

    selected_indices = []
    selected_image_ids = set()

    remaining_indices = list(range(len(all_captions)))

    while remaining_indices and len(selected_indices) < TOP_K_FINAL:

        best_index = None
        best_mmr_score = -float("inf")

        for idx in remaining_indices:

            image_id = image_ids[idx]

            # Une seule caption par image source
            if image_id in selected_image_ids:
                continue

            relevance = float(relevance_scores[idx])

            if not selected_indices:
                redundancy = 0.0

            else:
                selected_embeddings = text_embeddings[selected_indices]

                similarities = (
                    text_embeddings[idx] @ selected_embeddings.T
                )

                redundancy = float(np.max(similarities))

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

        selected_image_ids.add(image_ids[best_index])

        remaining_indices.remove(best_index)

    results = []

    for idx in selected_indices:

        results.append({
            "image_id": image_ids[idx],
            "caption": all_captions[idx],
            "relevance_score": float(relevance_scores[idx])
        })

    return results


# ============================================================
# 8. EVALUATION DES PARAMETRES MMR
# ============================================================

def evaluate_mmr_parameters():

    results = []

    total_queries = len(val_ids)

    for query_index, query_id in enumerate(val_ids):

        print(
            f"\nQuery {query_index + 1}/{total_queries}: {query_id}"
        )

        query_embedding = val_embeddings[query_index]

        # Les mêmes candidats sont utilisés pour chaque Lambda
        candidates = retrieve_candidates(
            query_embedding,
            TOP_K_FAISS
        )

        if not candidates:
            print("No candidates found.")
            continue

        query_results = {
            "query_id": query_id,
            "number_of_candidates": len(candidates),
            "lambda_results": {}
        }

        for lambda_mmr in MMR_LAMBDAS:

            print(f"Testing Lambda = {lambda_mmr}")

            selected = apply_mmr(
                query_embedding,
                candidates,
                lambda_mmr
            )

            query_results["lambda_results"][str(lambda_mmr)] = selected

        results.append(query_results)

    return results


# ============================================================
# 9. SAUVEGARDE ET ANALYSE
# ============================================================

def main():

    print("\nStarting MMR parameter evaluation...")

    results = evaluate_mmr_parameters()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nEvaluation completed.")
    print("Results saved to:", OUTPUT_FILE)

    print("\n========== MMR PARAMETER ANALYSIS ==========")

    for lambda_mmr in MMR_LAMBDAS:

        key = str(lambda_mmr)

        relevance_values = []
        unique_counts = []
        selected_count = 0

        for query in results:

            selected = query["lambda_results"].get(key, [])

            selected_count += len(selected)

            relevance_values.extend(
                item["relevance_score"]
                for item in selected
            )

            unique_counts.append(
                len({
                    item["image_id"]
                    for item in selected
                })
            )

        mean_relevance = (
            float(np.mean(relevance_values))
            if relevance_values else 0.0
        )

        mean_unique = (
            float(np.mean(unique_counts))
            if unique_counts else 0.0
        )

        print(f"\nLambda = {lambda_mmr}")
        print("Mean relevance:", round(mean_relevance, 4))
        print("Mean unique images:", round(mean_unique, 2))
        print("Total selected captions:", selected_count)

    print("\n============================================")


if __name__ == "__main__":
    main()
