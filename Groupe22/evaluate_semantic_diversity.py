import json
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

INPUT_FILE = EMBEDDINGS_DIR / "mmr_parameter_results.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

MMR_LAMBDAS = [0.5, 0.7, 0.9]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FILE = EMBEDDINGS_DIR / "semantic_diversity_results.json"

print("=" * 55)
print("       SEMANTIC DIVERSITY EVALUATION")
print("=" * 55)

print(f"Device: {DEVICE}")
print(f"Input file: {INPUT_FILE}")
print(f"Lambda values: {MMR_LAMBDAS}")
# Load MMR parameter results
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

print("\nData loaded successfully!")
print(f"Number of queries: {len(results)}")

# Verify Lambda values
for query in results:
    for lambda_value in MMR_LAMBDAS:
        key = str(lambda_value)

        if key not in query["lambda_results"]:
            raise ValueError(
                f"Missing Lambda {key} in query {query['query_id']}"
            )

print("Lambda values verification: PASSED")

# Verify selected results
for query in results:
    for lambda_value in MMR_LAMBDAS:
        key = str(lambda_value)
        selected = query["lambda_results"][key]

        if len(selected) != 5:
            raise ValueError(
                f"Expected 5 results for query {query['query_id']}, "
                f"Lambda {key}"
            )

print("Selected results verification: PASSED")
# Load CLIP model and tokenizer
print("\nLoading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("CLIP model loaded successfully!")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
# Encode captions into CLIP embeddings
def encode_captions(captions):
    if not captions:
        return np.empty((0, 512), dtype=np.float32)

    all_embeddings = []

    for start in range(0, len(captions), 32):
        batch = captions[start:start + 32]

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

            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

        all_embeddings.append(
            text_features.cpu().numpy().astype(np.float32)
        )

    return np.concatenate(all_embeddings, axis=0)

# Calculate semantic diversity using cosine similarity
def calculate_semantic_diversity(captions):
    if len(captions) < 2:
        return 0.0

    embeddings = encode_captions(captions)

    # Calculate pairwise cosine similarity
    similarity_matrix = embeddings @ embeddings.T

    # Extract upper triangle without diagonal
    upper_indices = np.triu_indices(
        len(captions),
        k=1
    )

    pairwise_similarities = similarity_matrix[upper_indices]

    mean_similarity = float(
        np.mean(pairwise_similarities)
    )

    # Diversity = 1 - mean similarity
    diversity_score = 1.0 - mean_similarity

    return {
        "mean_similarity": mean_similarity,
        "diversity_score": diversity_score
    }

# Evaluate semantic diversity for all queries
def evaluate_all_queries():

    evaluation_results = []

    for query_index, query in enumerate(results, start=1):

        query_id = query["query_id"]

        query_result = {
            "query_id": query_id,
            "lambda_results": {}
        }

        for lambda_value in MMR_LAMBDAS:

            key = str(lambda_value)

            selected = query["lambda_results"][key]

            captions = [
                item["caption"]
                for item in selected
            ]

            diversity = calculate_semantic_diversity(captions)

            query_result["lambda_results"][key] = {
                "mean_similarity": diversity["mean_similarity"],
                "diversity_score": diversity["diversity_score"]
            }

        evaluation_results.append(query_result)

        if query_index % 10 == 0:
            print(
                f"Processed {query_index}/{len(results)} queries"
            )

    return evaluation_results

# Calculate average semantic diversity for each Lambda
def summarize_results(evaluation_results):

    summary = {}

    print("\n" + "=" * 55)
    print("SEMANTIC DIVERSITY COMPARISON")
    print("=" * 55)

    for lambda_value in MMR_LAMBDAS:

        key = str(lambda_value)

        similarities = []
        diversity_scores = []

        for query in evaluation_results:

            metrics = query["lambda_results"][key]

            similarities.append(
                metrics["mean_similarity"]
            )

            diversity_scores.append(
                metrics["diversity_score"]
            )

        mean_similarity = float(np.mean(similarities))
        mean_diversity = float(np.mean(diversity_scores))

        summary[key] = {
            "mean_similarity": mean_similarity,
            "mean_diversity": mean_diversity
        }

        print(f"\nLambda: {key}")
        print(f"Mean Similarity: {mean_similarity:.4f}")
        print(f"Mean Diversity: {mean_diversity:.4f}")

    return summary
# Main execution
def main():

    print("\nStarting semantic diversity evaluation...")

    evaluation_results = evaluate_all_queries()

    summary = summarize_results(evaluation_results)

    # Prepare final output
    final_results = {
        "number_of_queries": len(evaluation_results),
        "lambda_values": MMR_LAMBDAS,
        "summary": summary,
        "detailed_results": evaluation_results
    }

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            final_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()    