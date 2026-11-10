# ============================================================
# 1. IMPORTS
# ============================================================

import json
from pathlib import Path

import faiss
import numpy as np
import torch

from transformers import CLIPModel, CLIPTokenizer


# ============================================================
# 2. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"
SPLITS_DIR = DATASET_DIR / "dataset" / "splits"

FAISS_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"

VAL_IDS_FILE = EMBEDDINGS_DIR / "val2017_ids.json"
VAL_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "val2017_embeddings.npy"

KB_FILE = DATASET_DIR / "knowledge_base_train.json"

VAL_ANNOTATIONS_FILE = (
    SPLITS_DIR / "validation" / "validation.json"
)

CALIBRATION_FILE = (
    EMBEDDINGS_DIR / "crag_calibration_results.json"
)

RERANKING_FILE = (
    EMBEDDINGS_DIR / "crag_reranking_results.json"
)

MMR_FILE = (
    EMBEDDINGS_DIR / "crag_diversification_results.json"
)

OUTPUT_FILE = (
    EMBEDDINGS_DIR / "evaluation_results.json"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K_FAISS = 20
TOP_K_FINAL = 5

BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 3. BASIC CONFIGURATION CHECK
# ============================================================

print("=" * 55)
print("       RETRIEVAL EVALUATION PIPELINE")
print("=" * 55)

print("Device:", DEVICE)
print("FAISS Top-K:", TOP_K_FAISS)
print("Final Top-K:", TOP_K_FINAL)

print("\nConfiguration loaded successfully.")

# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading project data...")

# FAISS index
index = faiss.read_index(str(FAISS_FILE))

# Image IDs
with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
    train_ids = json.load(f)

with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
    val_ids = json.load(f)

# Validation image embeddings
val_embeddings = np.load(VAL_EMBEDDINGS_FILE)

# Knowledge base
with open(KB_FILE, "r", encoding="utf-8") as f:
    knowledge_base_list = json.load(f)

# Validation annotations
with open(VAL_ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    validation_data = json.load(f)

# Previous experiment results
with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
    calibration_results = json.load(f)

with open(RERANKING_FILE, "r", encoding="utf-8") as f:
    reranking_results = json.load(f)

with open(MMR_FILE, "r", encoding="utf-8") as f:
    mmr_results = json.load(f)


# ============================================================
# 5. NORMALIZE IMAGE IDS
# ============================================================

def normalize_image_id(image_id):
    return f"{int(image_id):012d}"


train_ids = [
    normalize_image_id(image_id)
    for image_id in train_ids
]

val_ids = [
    normalize_image_id(image_id)
    for image_id in val_ids
]

knowledge_base = {
    normalize_image_id(item["image_id"]): item
    for item in knowledge_base_list
}


# ============================================================
# 6. BUILD VALIDATION GROUND DATA
# ============================================================

validation_images = {
    normalize_image_id(image["id"]): image
    for image in validation_data["images"]
}

reference_captions = {}

for annotation in validation_data["annotations"]:

    image_id = normalize_image_id(annotation["image_id"])

    reference_captions.setdefault(image_id, []).append(
        annotation["caption"]
    )


# ============================================================
# 7. DATA VALIDATION
# ============================================================

print("\nChecking data consistency...")

assert index.ntotal == len(train_ids), \
    "FAISS index and train IDs do not match."

assert len(val_ids) == len(val_embeddings), \
    "Validation IDs and embeddings do not match."

assert len(knowledge_base) == len(train_ids), \
    "Knowledge base and train IDs do not match."

assert len(validation_images) == len(val_ids), \
    "Validation image count mismatch."

assert set(train_ids).isdisjoint(set(val_ids)), \
    "Data leakage detected: train/validation overlap."

assert all(
    len(reference_captions.get(image_id, [])) == 5
    for image_id in val_ids
), "Some validation images do not have exactly 5 captions."

print("FAISS vectors:", index.ntotal)
print("Training images:", len(train_ids))
print("Validation images:", len(val_ids))
print("Knowledge base:", len(knowledge_base))
print("Validation annotations:", len(validation_data["annotations"]))

print("\nAll data checks passed.")

# ============================================================
# 8. LOAD CLIP MODEL
# ============================================================

print("\nLoading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)

tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("CLIP model loaded on:", DEVICE)


# ============================================================
# 9. ENCODE TEXT CAPTIONS
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

    return np.concatenate(
        all_embeddings,
        axis=0
    ).astype(np.float32)


# ============================================================
# 10. COMPUTE CLIP IMAGE-TEXT SIMILARITY
# ============================================================

def compute_clip_scores(query_embedding, captions):

    text_embeddings = encode_captions(captions)

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        raise ValueError("Query embedding is zero.")

    query_embedding /= query_norm

    similarities = (
        text_embeddings @ query_embedding.T
    ).flatten()

    return similarities

# ============================================================
# 11. BASELINE RETRIEVAL WITH FAISS
# ============================================================

def baseline_retrieval(query_embedding, top_k=TOP_K_FINAL):

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        raise ValueError("Query embedding is zero.")

    query_embedding /= query_norm

    # Retrieve candidates from FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    validation_set = set(val_ids)

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        image_id = train_ids[int(idx)]

        # Exclude validation images
        if image_id in validation_set:
            continue

        entry = knowledge_base.get(image_id)

        if entry is None:
            continue

        captions = entry.get("captions", [])

        if not captions:
            continue

        results.append({
            "image_id": image_id,
            "faiss_score": float(score),
            "captions": captions
        })

    return results

# ============================================================
# 12. CRAG + RERANKING
# ============================================================

def crag_reranking(query_embedding):

    # Retrieve 20 candidates using FAISS
    candidates = baseline_retrieval(
        query_embedding,
        TOP_K_FAISS
    )

    if not candidates:
        return []

    # Collect all candidate captions
    all_captions = []

    for candidate in candidates:
        all_captions.extend(candidate["captions"])

    # Compute CLIP similarity scores
    similarities = compute_clip_scores(
        query_embedding,
        all_captions
    )

    position = 0

    for candidate in candidates:

        number = len(candidate["captions"])

        candidate_scores = similarities[
            position:position + number
        ]

        best_index = int(np.argmax(candidate_scores))

        candidate["best_caption"] = (
            candidate["captions"][best_index]
        )

        candidate["clip_score"] = float(
            candidate_scores[best_index]
        )

        position += number

    # Sort candidates by CLIP score
    candidates.sort(
        key=lambda item: item["clip_score"],
        reverse=True
    )

    return candidates[:TOP_K_FINAL]

# ============================================================
# 13. CRAG CLASSIFICATION
# ============================================================

LOWER_THRESHOLD = 0.3170
UPPER_THRESHOLD = 0.3402


def classify_query(score):

    if score >= UPPER_THRESHOLD:
        return "CORRECT"

    elif score >= LOWER_THRESHOLD:
        return "AMBIGUOUS"

    else:
        return "INCORRECT"


# ============================================================
# 14. CRAG ADAPTIVE RETRIEVAL
# ============================================================

def crag_adaptive_retrieval(query_embedding, initial_score):

    classification = classify_query(initial_score)

    if classification == "CORRECT":
        top_k = 20
        action = "KEEP_INITIAL_RESULTS"

    elif classification == "AMBIGUOUS":
        top_k = 50
        action = "EXPAND_RETRIEVAL"

    else:
        top_k = 50
        action = "FALLBACK_EXPANDED_RETRIEVAL"

    # Apply the selected retrieval strategy
    candidates = baseline_retrieval(
        query_embedding,
        top_k
    )

    return {
        "classification": classification,
        "action": action,
        "candidates": candidates
    }

# ============================================================
# 15. MMR DIVERSIFICATION
# ============================================================

LAMBDA_MMR = 0.7


def mmr_diversification(query_embedding, candidates):

    if not candidates:
        return []

    # Collect captions and their source image IDs
    all_captions = []
    caption_image_ids = []

    for candidate in candidates:

        for caption in candidate["captions"]:

            all_captions.append(caption)
            caption_image_ids.append(candidate["image_id"])

    # Encode all captions
    text_embeddings = encode_captions(all_captions)

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    query_embedding /= np.linalg.norm(query_embedding)

    # Relevance scores
    relevance_scores = (
        text_embeddings @ query_embedding.T
    ).flatten()

    selected_indices = []
    selected_image_ids = set()

    remaining_indices = list(range(len(all_captions)))

    while remaining_indices and len(selected_indices) < TOP_K_FINAL:

        best_index = None
        best_mmr_score = -float("inf")

        for idx in remaining_indices:

            image_id = caption_image_ids[idx]

            # One caption per source image
            if image_id in selected_image_ids:
                continue

            relevance = float(relevance_scores[idx])

            if selected_indices:

                selected_embeddings = text_embeddings[
                    selected_indices
                ]

                redundancy = float(
                    np.max(
                        text_embeddings[idx] @ selected_embeddings.T
                    )
                )

            else:
                redundancy = 0.0

            mmr_score = (
                LAMBDA_MMR * relevance
                - (1 - LAMBDA_MMR) * redundancy
            )

            if mmr_score > best_mmr_score:

                best_mmr_score = mmr_score
                best_index = idx

        if best_index is None:
            break

        selected_indices.append(best_index)

        selected_image_ids.add(
            caption_image_ids[best_index]
        )

        remaining_indices.remove(best_index)

    # Prepare final results
    results = []

    for idx in selected_indices:

        results.append({
            "image_id": caption_image_ids[idx],
            "caption": all_captions[idx],
            "relevance_score": float(
                relevance_scores[idx]
            )
        })

    return results

# ============================================================
# 16. EVALUATE ALL VALIDATION QUERIES
# ============================================================

def evaluate_all_queries():

    evaluation_results = []

    print("\nStarting evaluation on validation set...")

    for query_index, query_id in enumerate(val_ids):

        print(
            f"\nQuery {query_index + 1}/{len(val_ids)}:",
            query_id
        )

        query_embedding = val_embeddings[query_index]

        # ----------------------------------------------------
        # A. BASELINE: FAISS TOP 5
        # ----------------------------------------------------

        baseline_results = baseline_retrieval(
            query_embedding,
            TOP_K_FINAL
        )

        baseline_output = [
            {
                "image_id": item["image_id"],
                "faiss_score": item["faiss_score"]
            }
            for item in baseline_results
        ]

        # ----------------------------------------------------
        # B. INITIAL FAISS RETRIEVAL FOR CRAG
        # ----------------------------------------------------

        initial_candidates = baseline_retrieval(
            query_embedding,
            TOP_K_FAISS
        )

        initial_captions = []

        for candidate in initial_candidates:
            initial_captions.extend(candidate["captions"])

        if not initial_captions:
            print("No candidates found. Skipping query.")
            continue

        initial_scores = compute_clip_scores(
            query_embedding,
            initial_captions
        )

        initial_score = float(np.max(initial_scores))

        # ----------------------------------------------------
        # C. CRAG CLASSIFICATION AND ADAPTIVE RETRIEVAL
        # ----------------------------------------------------

        crag_result = crag_adaptive_retrieval(
            query_embedding,
            initial_score
        )

        candidates = crag_result["candidates"]

        # ----------------------------------------------------
        # D. CLIP RERANKING
        # ----------------------------------------------------

        all_captions = []

        for candidate in candidates:
            all_captions.extend(candidate["captions"])

        clip_scores = compute_clip_scores(
            query_embedding,
            all_captions
        )

        position = 0

        for candidate in candidates:

            count = len(candidate["captions"])

            candidate_scores = clip_scores[
                position:position + count
            ]

            best_index = int(np.argmax(candidate_scores))

            candidate["best_caption"] = (
                candidate["captions"][best_index]
            )

            candidate["clip_score"] = float(
                candidate_scores[best_index]
            )

            position += count

        reranked_candidates = sorted(
            candidates,
            key=lambda item: item["clip_score"],
            reverse=True
        )

        reranking_output = [
            {
                "image_id": item["image_id"],
                "caption": item["best_caption"],
                "clip_score": item["clip_score"]
            }
            for item in reranked_candidates[:TOP_K_FINAL]
        ]

        # ----------------------------------------------------
        # E. MMR DIVERSIFICATION
        # ----------------------------------------------------

        mmr_output = mmr_diversification(
            query_embedding,
            candidates
        )

        # ----------------------------------------------------
        # F. SAVE QUERY RESULTS
        # ----------------------------------------------------

        evaluation_results.append({

            "query_id": query_id,

            "initial_score": initial_score,

            "classification": crag_result["classification"],

            "action": crag_result["action"],

            "baseline": baseline_output,

            "reranking": reranking_output,

            "mmr": mmr_output

        })

    return evaluation_results

# ============================================================
# 17. SAVE EVALUATION RESULTS
# ============================================================

def main():

    print("\n" + "=" * 55)
    print("       STARTING FULL EVALUATION")
    print("=" * 55)

    results = evaluate_all_queries()

    # Save results to JSON
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("             EVALUATION SUMMARY")
    print("=" * 55)

    print("Processed queries:", len(results))

    print(
        "Baseline results:",
        sum(len(x["baseline"]) for x in results)
    )

    print(
        "Reranking results:",
        sum(len(x["reranking"]) for x in results)
    )

    print(
        "MMR results:",
        sum(len(x["mmr"]) for x in results)
    )

    classifications = {}

    for result in results:

        label = result["classification"]

        classifications[label] = (
            classifications.get(label, 0) + 1
        )

    print("\nCRAG classifications:")

    for label, count in classifications.items():

        print(f"{label}: {count}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nEvaluation completed.")


# ============================================================
# 18. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()