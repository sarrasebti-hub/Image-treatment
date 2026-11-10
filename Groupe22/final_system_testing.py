
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

FAISS_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"
TEST_IDS_FILE = EMBEDDINGS_DIR / "test2017_ids.json"
TEST_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "test2017_embeddings.npy"

KB_FILE = DATASET_DIR / "knowledge_base_train.json"
TEST_JSON = DATASET_DIR / "dataset" / "splits" / "test" / "test_cleaned.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_FAISS = 20
TOP_K_EXPANDED = 50
TOP_K_FINAL = 5

LAMBDA_MMR = 0.9

THRESHOLD_CORRECT = 0.3402
THRESHOLD_AMBIGUOUS = 0.3170

BATCH_SIZE = 64

OUTPUT_FILE = EMBEDDINGS_DIR / "final_system_test_results.json"
REPORT_FILE = DATASET_DIR / "FINAL_SYSTEM_TEST_REPORT.md"


# ============================================================
# 2. UTILITIES
# ============================================================

def normalize_id(image_id):
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


def encode_captions(captions, model, tokenizer):

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

    return np.vstack(embeddings).astype("float32")


# ============================================================
# 3. MMR
# ============================================================

def mmr_selection(
    captions_data,
    query_embedding,
    caption_embeddings,
    final_k=5,
    lambda_mmr=0.9
):

    relevance_scores = caption_embeddings @ query_embedding

    selected_indices = []
    selected_image_ids = set()

    remaining_indices = set(range(len(captions_data)))

    while len(selected_indices) < final_k and remaining_indices:

        best_index = None
        best_score = -float("inf")

        for idx in remaining_indices:

            image_id = captions_data[idx]["image_id"]

            if image_id in selected_image_ids:
                continue

            relevance = float(relevance_scores[idx])

            if selected_indices:

                similarities = (
                    caption_embeddings[idx]
                    @ caption_embeddings[selected_indices].T
                )

                redundancy = float(np.max(similarities))

            else:
                redundancy = 0.0

            mmr_score = (
                lambda_mmr * relevance
                - (1 - lambda_mmr) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = idx

        if best_index is None:
            break

        selected_indices.append(best_index)

        selected_image_ids.add(
            captions_data[best_index]["image_id"]
        )

        remaining_indices.remove(best_index)

    results = []

    for idx in selected_indices:

        results.append({
            "image_id": captions_data[idx]["image_id"],
            "caption": captions_data[idx]["caption"],
            "relevance_score": round(
                float(relevance_scores[idx]), 4
            )
        })

    return results


# ============================================================
# 4. TEXT EVALUATION
# ============================================================

def evaluate_captions(predictions, references):

    if not predictions or not references:
        return {
            "bleu": 0.0,
            "rouge_l": 0.0
        }

    smoothing = SmoothingFunction().method1

    bleu_scores = []
    rouge_scores = []

    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    for prediction in predictions:

        tokens = prediction.lower().split()

        reference_tokens = [
            reference.lower().split()
            for reference in references
        ]

        bleu = sentence_bleu(
            reference_tokens,
            tokens,
            smoothing_function=smoothing
        )

        bleu_scores.append(bleu)

        rouge_values = [
            rouge.score(reference, prediction)["rougeL"].fmeasure
            for reference in references
        ]

        rouge_scores.append(max(rouge_values))

    return {
        "bleu": float(np.mean(bleu_scores)),
        "rouge_l": float(np.mean(rouge_scores))
    }


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 65)
print("             FINAL SYSTEM TESTING")
print("=" * 65)

print("\n[1/6] Loading data...")

index = faiss.read_index(str(FAISS_FILE))

train_ids = [
    normalize_id(x)
    for x in load_json(TRAIN_IDS_FILE)
]

test_ids = [
    normalize_id(x)
    for x in load_json(TEST_IDS_FILE)
]

test_embeddings = np.load(
    TEST_EMBEDDINGS_FILE
).astype("float32")

knowledge_base_list = load_json(KB_FILE)

knowledge_base = {
    normalize_id(item["image_id"]): item
    for item in knowledge_base_list
}

test_data = load_json(TEST_JSON)

test_references = {}

for annotation in test_data["annotations"]:

    image_id = normalize_id(annotation["image_id"])

    test_references.setdefault(image_id, []).append(
        annotation["caption"]
    )


# ============================================================
# 6. DATA VALIDATION
# ============================================================

print("\n[2/6] Validating data...")

if index.ntotal != len(train_ids):
    raise ValueError("FAISS index and train IDs mismatch.")

if len(test_ids) != len(test_embeddings):
    raise ValueError("Test IDs and embeddings mismatch.")

if len(set(test_ids)) != len(test_ids):
    raise ValueError("Duplicate test IDs.")

if set(test_ids) & set(train_ids):
    raise ValueError("Data leakage: Test overlaps Training.")

if len(test_ids) != 100:
    raise ValueError("Expected 100 test images.")

for image_id in test_ids:

    if image_id not in test_references:
        raise ValueError(
            f"Missing reference captions for {image_id}"
        )

    if len(test_references[image_id]) != 5:
        raise ValueError(
            f"Expected 5 references for {image_id}"
        )

print("FAISS:", index.ntotal)
print("Test images:", len(test_ids))
print("Test embeddings:", test_embeddings.shape)
print("Reference captions:", sum(
    len(test_references[x]) for x in test_ids
))
print("No train/test overlap.")


# ============================================================
# 7. LOAD CLIP
# ============================================================

print("\n[3/6] Loading CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("CLIP loaded on:", DEVICE)


# ============================================================
# 8. FINAL RETRIEVAL
# ============================================================

print("\n[4/6] Running CRAG + FAISS + MMR...")

results = []

classification_counts = {
    "CORRECT": 0,
    "AMBIGUOUS": 0,
    "INCORRECT": 0
}

for query_number, query_id in enumerate(test_ids, start=1):

    query = test_embeddings[query_number - 1].copy()

    norm = np.linalg.norm(query)

    if norm == 0:
        raise ValueError(f"Zero embedding: {query_id}")

    query = (query / norm).astype("float32")

    # Initial retrieval with 20 candidates
    initial_scores, initial_indices = index.search(
        query.reshape(1, -1),
        TOP_K_FAISS
    )

    initial_candidates = []

    for idx in initial_indices[0]:

        if idx < 0:
            continue

        image_id = train_ids[int(idx)]

        if image_id not in knowledge_base:
            continue

        initial_candidates.append(image_id)

    initial_captions = []

    for image_id in initial_candidates:

        item = knowledge_base[image_id]

        for caption in item.get("captions", []):

            initial_captions.append({
                "image_id": image_id,
                "caption": caption
            })

    if not initial_captions:
        continue

    initial_text_embeddings = encode_captions(
        [x["caption"] for x in initial_captions],
        model,
        tokenizer
    )

    initial_relevance = initial_text_embeddings @ query

    best_initial_score = max(
        float(np.max(initial_relevance[
            [
                i for i, item in enumerate(initial_captions)
                if item["image_id"] == image_id
            ]
        ]))
        for image_id in initial_candidates
    )

    classification = classify_retrieval(
        best_initial_score
    )

    classification_counts[classification] += 1

    # CRAG action
    if classification == "CORRECT":
        top_k = TOP_K_FAISS
        action = "RETRIEVE_TOP_20"

    elif classification == "AMBIGUOUS":
        top_k = TOP_K_EXPANDED
        action = "EXPANDED_RETRIEVAL_TOP_50"

    else:
        top_k = TOP_K_EXPANDED
        action = "FALLBACK_RETRIEVAL_TOP_50"

    # Final candidate retrieval
    _, candidate_indices = index.search(
        query.reshape(1, -1),
        top_k
    )

    candidate_ids = []

    for idx in candidate_indices[0]:

        if idx < 0:
            continue

        image_id = train_ids[int(idx)]

        if image_id in knowledge_base:
            candidate_ids.append(image_id)

    captions_data = []

    for image_id in candidate_ids:

        for caption in knowledge_base[image_id].get("captions", []):

            captions_data.append({
                "image_id": image_id,
                "caption": caption
            })

    if not captions_data:
        continue

    caption_embeddings = encode_captions(
        [x["caption"] for x in captions_data],
        model,
        tokenizer
    )

    selected = mmr_selection(
        captions_data,
        query,
        caption_embeddings,
        final_k=TOP_K_FINAL,
        lambda_mmr=LAMBDA_MMR
    )

    predictions = [
        item["caption"]
        for item in selected
    ]

    references = test_references[query_id]

    metrics = evaluate_captions(
        predictions,
        references
    )

    results.append({
        "query_image_id": query_id,
        "initial_score": round(best_initial_score, 4),
        "classification": classification,
        "action": action,
        "candidate_count": len(candidate_ids),
        "lambda_mmr": LAMBDA_MMR,
        "selected_captions": selected,
        "reference_captions": references,
        "metrics": metrics
    })

    print(
        f"{query_number}/100 | {query_id} | "
        f"{classification} | "
        f"BLEU={metrics['bleu']:.4f} | "
        f"ROUGE-L={metrics['rouge_l']:.4f}"
    )


# ============================================================
# 9. SUMMARY
# ============================================================

if not results:
    raise RuntimeError("No test results were generated.")

mean_bleu = np.mean([
    x["metrics"]["bleu"] for x in results
])

mean_rouge = np.mean([
    x["metrics"]["rouge_l"] for x in results
])

mean_clip_relevance = np.mean([
    np.mean([
        item["relevance_score"]
        for item in x["selected_captions"]
    ])
    for x in results
])

summary = {
    "test_queries": len(results),
    "lambda_mmr": LAMBDA_MMR,
    "threshold_correct": THRESHOLD_CORRECT,
    "threshold_ambiguous": THRESHOLD_AMBIGUOUS,
    "classification_counts": classification_counts,
    "mean_bleu": float(mean_bleu),
    "mean_rouge_l": float(mean_rouge),
    "mean_clip_relevance": float(mean_clip_relevance)
}

output = {
    "summary": summary,
    "results": results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)


# ============================================================
# 10. REPORT
# ============================================================

report = f"""# Final System Test Report

## Test Configuration

- Test queries: {len(results)}
- Training images: {len(train_ids)}
- FAISS candidates: 20 / 50 according to CRAG
- Final captions: {TOP_K_FINAL}
- MMR lambda: {LAMBDA_MMR}
- Correct threshold: {THRESHOLD_CORRECT}
- Ambiguous threshold: {THRESHOLD_AMBIGUOUS}

## CRAG Classification

- CORRECT: {classification_counts['CORRECT']}
- AMBIGUOUS: {classification_counts['AMBIGUOUS']}
- INCORRECT: {classification_counts['INCORRECT']}

## Evaluation Metrics

- Mean BLEU: {mean_bleu:.4f}
- Mean ROUGE-L: {mean_rouge:.4f}
- Mean CLIP relevance: {mean_clip_relevance:.4f}

## Methodological Limitations

BLEU and ROUGE-L measure textual overlap with reference captions.
They do not independently establish semantic correctness.

CRAG classifications use thresholds calibrated on Validation.
They are operational decisions, not verified accuracy labels.

The Test set was not used to adjust thresholds or MMR lambda.
"""

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + "=" * 65)
print("FINAL TEST COMPLETED")
print("=" * 65)

print("Test queries:", len(results))
print("Mean BLEU:", round(mean_bleu, 4))
print("Mean ROUGE-L:", round(mean_rouge, 4))
print("Mean CLIP relevance:", round(mean_clip_relevance, 4))

print("\nResults:", OUTPUT_FILE)
print("Report:", REPORT_FILE)
