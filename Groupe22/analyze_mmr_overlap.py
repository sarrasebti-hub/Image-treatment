import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    BASE_DIR / "dataset" / "embeddings"
    / "mmr_parameter_results.json"
)

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = [
    ("0.5", "0.7"),
    ("0.5", "0.9"),
    ("0.7", "0.9")
]

for lambda_a, lambda_b in pairs:
    similarities = []

    for query in data:
        results_a = query["lambda_results"][lambda_a]
        results_b = query["lambda_results"][lambda_b]

        set_a = {r["image_id"] for r in results_a}
        set_b = {r["image_id"] for r in results_b}

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        jaccard = intersection / union if union else 0
        similarities.append(jaccard)

    mean_jaccard = sum(similarities) / len(similarities)

    print(f"\nLambda {lambda_a} vs Lambda {lambda_b}")
    print("Mean Jaccard Similarity:", round(mean_jaccard, 4))