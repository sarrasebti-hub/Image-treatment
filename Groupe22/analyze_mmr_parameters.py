import json
from pathlib import Path
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "dataset" / "embeddings"

INPUT_FILE = EMBEDDINGS_DIR / "mmr_parameter_results.json"

# MMR parameters
MMR_LAMBDAS = [0.5, 0.7, 0.9]

print("=" * 55)
print("       MMR PARAMETERS COMPARATIVE ANALYSIS")
print("=" * 55)

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
# Calculate mean relevance for each Lambda
print("\n" + "=" * 55)
print("MEAN RELEVANCE COMPARISON")
print("=" * 55)

mean_relevance = {}

for lambda_value in MMR_LAMBDAS:
    key = str(lambda_value)
    relevance_scores = []

    for query in results:
        selected = query["lambda_results"][key]

        for item in selected:
            relevance_scores.append(item["relevance_score"])

    mean_score = float(np.mean(relevance_scores))
    mean_relevance[key] = mean_score

    print(f"Lambda {key}: Mean Relevance = {mean_score:.4f}")

    # Calculate unique images for each Lambda
print("\n" + "=" * 55)
print("UNIQUE IMAGES COMPARISON")
print("=" * 55)

mean_unique_images = {}

for lambda_value in MMR_LAMBDAS:
    key = str(lambda_value)
    unique_counts = []

    for query in results:
        selected = query["lambda_results"][key]

        unique_ids = {
            item["image_id"] for item in selected
        }

        unique_counts.append(len(unique_ids))

    mean_unique = float(np.mean(unique_counts))
    mean_unique_images[key] = mean_unique

    print(
        f"Lambda {key}: "
        f"Mean Unique Images = {mean_unique:.2f}"
    )


 # Compare selected images between Lambda values
print("\n" + "=" * 55)
print("IMAGE OVERLAP COMPARISON")
print("=" * 55)

lambda_pairs = [
    ("0.5", "0.7"),
    ("0.5", "0.9"),
    ("0.7", "0.9")
]

for lambda_a, lambda_b in lambda_pairs:
    overlap_counts = []

    for query in results:
        images_a = {
            item["image_id"]
            for item in query["lambda_results"][lambda_a]
        }

        images_b = {
            item["image_id"]
            for item in query["lambda_results"][lambda_b]
        }

        common_images = images_a.intersection(images_b)
        overlap_counts.append(len(common_images))

    mean_overlap = float(np.mean(overlap_counts))

    print(
        f"Lambda {lambda_a} vs {lambda_b}: "
        f"Mean shared images = {mean_overlap:.2f}/5"
    )

 # Identify the best Lambda based on mean relevance
best_lambda = max(
    mean_relevance,
    key=mean_relevance.get
)

print("\n" + "=" * 55)
print("BEST LAMBDA")
print("=" * 55)

print(f"Best Lambda: {best_lambda}")
print(f"Mean Relevance: {mean_relevance[best_lambda]:.4f}")

# Prepare analysis summary
summary = {
    "mean_relevance": mean_relevance,
    "mean_unique_images": mean_unique_images,
    "best_lambda": best_lambda,
    "image_overlap": {}
}

for lambda_a, lambda_b in lambda_pairs:
    overlap_counts = []

    for query in results:
        images_a = {
            item["image_id"]
            for item in query["lambda_results"][lambda_a]
        }

        images_b = {
            item["image_id"]
            for item in query["lambda_results"][lambda_b]
        }

        overlap_counts.append(
            len(images_a.intersection(images_b))
        )

    summary["image_overlap"][
        f"{lambda_a}_vs_{lambda_b}"
    ] = float(np.mean(overlap_counts))

# Save analysis summary
OUTPUT_FILE = EMBEDDINGS_DIR / "mmr_analysis_summary.json"

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)

print("\n" + "=" * 55)
print("ANALYSIS COMPLETED")
print("=" * 55)

print(f"Results saved to: {OUTPUT_FILE}")          