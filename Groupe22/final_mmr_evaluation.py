import json
from pathlib import Path
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

RELEVANCE_FILE = EMBEDDINGS_DIR / "mmr_parameter_results.json"
DIVERSITY_FILE = EMBEDDINGS_DIR / "semantic_diversity_results.json"

OUTPUT_FILE = EMBEDDINGS_DIR / "final_mmr_evaluation.json"

# MMR parameters
MMR_LAMBDAS = [0.5, 0.7, 0.9]

print("=" * 60)
print("          FINAL MMR EVALUATION")
print("=" * 60)

print(f"Lambda values: {MMR_LAMBDAS}")
print(f"Device-independent analysis")
# Load relevance results
with open(RELEVANCE_FILE, "r", encoding="utf-8") as file:
    relevance_data = json.load(file)

# Load diversity results
with open(DIVERSITY_FILE, "r", encoding="utf-8") as file:
    diversity_data = json.load(file)

print("\nData loaded successfully!")

print(
    f"Relevance queries: {len(relevance_data)}"
)

print(
    f"Diversity queries: {diversity_data['number_of_queries']}"
)

# Verify query counts
if len(relevance_data) != diversity_data["number_of_queries"]:
    raise ValueError("Query counts do not match!")

print("Data verification: PASSED")
# Calculate mean relevance for each Lambda
relevance_summary = {}

print("\n" + "=" * 60)
print("MEAN RELEVANCE")
print("=" * 60)

for lambda_value in MMR_LAMBDAS:

    key = str(lambda_value)
    scores = []

    for query in relevance_data:
        selected = query["lambda_results"][key]

        for item in selected:
            scores.append(item["relevance_score"])

    mean_score = float(np.mean(scores))

    relevance_summary[key] = mean_score

    print(
        f"Lambda {key}: "
        f"Mean Relevance = {mean_score:.4f}"
    )
    # Calculate mean diversity for each Lambda
diversity_summary = {}

print("\n" + "=" * 60)
print("MEAN SEMANTIC DIVERSITY")
print("=" * 60)

for lambda_value in MMR_LAMBDAS:

    key = str(lambda_value)

    diversity_scores = []

    for query in diversity_data["detailed_results"]:

        metrics = query["lambda_results"][key]

        diversity_scores.append(
            metrics["diversity_score"]
        )

    mean_diversity = float(np.mean(diversity_scores))

    diversity_summary[key] = mean_diversity

    print(
        f"Lambda {key}: "
        f"Mean Diversity = {mean_diversity:.4f}"
    )
    # Calculate combined score
combined_summary = {}

print("\n" + "=" * 60)
print("COMBINED SCORE")
print("=" * 60)

for lambda_value in MMR_LAMBDAS:

    key = str(lambda_value)

    relevance = relevance_summary[key]
    diversity = diversity_summary[key]

    combined_score = (
        relevance + diversity
    ) / 2

    combined_summary[key] = combined_score

    print(f"\nLambda: {key}")
    print(f"Relevance: {relevance:.4f}")
    print(f"Diversity: {diversity:.4f}")
    print(f"Combined Score: {combined_score:.4f}")
    # Identify the best Lambda
best_lambda = max(
    combined_summary,
    key=combined_summary.get
)

print("\n" + "=" * 60)
print("BEST LAMBDA")
print("=" * 60)

print(f"Best Lambda: {best_lambda}")
print(
    f"Combined Score: "
    f"{combined_summary[best_lambda]:.4f}"
)
# Prepare final evaluation report
final_report = {
    "number_of_queries": len(relevance_data),
    "lambda_values": MMR_LAMBDAS,
    "relevance": relevance_summary,
    "diversity": diversity_summary,
    "combined_score": combined_summary,
    "best_lambda": best_lambda,
    "best_combined_score": combined_summary[best_lambda]
}

# Save final report
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        final_report,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 60)
print("FINAL MMR EVALUATION COMPLETED")
print("=" * 60)

print(f"Results saved to: {OUTPUT_FILE}")
