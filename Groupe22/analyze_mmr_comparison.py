import json
from pathlib import Path
from itertools import combinations

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "dataset" / "embeddings" / "mmr_parameter_results.json"

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

lambdas = ["0.5", "0.7", "0.9"]

print("Number of queries:", len(data))

for lam in lambdas:
    scores = []
    image_ids = set()

    for query in data:
        results = query["lambda_results"][lam]

        for result in results:
            scores.append(result["relevance_score"])
            image_ids.add(result["image_id"])

    print(f"\nLambda = {lam}")
    print("Mean relevance:", round(sum(scores) / len(scores), 4))
    print("Unique images across all queries:", len(image_ids))
    print("Total results:", len(scores))