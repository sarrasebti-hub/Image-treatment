import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "dataset" / "embeddings"

OLD_FILE = EMBEDDINGS_DIR / "evaluation_results.json"
NEW_FILE = EMBEDDINGS_DIR / "crag_diversification_results.json"

with open(OLD_FILE, "r", encoding="utf-8") as f:
    old_data = json.load(f)

with open(NEW_FILE, "r", encoding="utf-8") as f:
    new_data = json.load(f)

old_map = {item["query_id"]: item for item in old_data}
new_map = {item["query_image_id"]: item for item in new_data}

common_ids = sorted(set(old_map) & set(new_map))

print("Old queries:", len(old_data))
print("New queries:", len(new_data))
print("Common queries:", len(common_ids))

old_scores = []
new_scores = []

old_images = set()
new_images = set()

overlap_scores = []

for query_id in common_ids:

    old_results = old_map[query_id]["mmr"]
    new_results = new_map[query_id]["selected_captions"]

    for result in old_results:
        old_scores.append(result["relevance_score"])
        old_images.add(result["image_id"])

    for result in new_results:
        new_scores.append(result["relevance_score"])
        new_images.add(result["image_id"])

    old_set = {r["image_id"] for r in old_results}
    new_set = {r["image_id"] for r in new_results}

    intersection = len(old_set & new_set)
    union = len(old_set | new_set)

    overlap_scores.append(intersection / union if union else 0)

print("\n========== FINAL MMR COMPARISON ==========")

print("\nLambda = 0.7")
print("Mean relevance:", round(sum(old_scores) / len(old_scores), 4))
print("Unique images:", len(old_images))

print("\nLambda = 0.9")
print("Mean relevance:", round(sum(new_scores) / len(new_scores), 4))
print("Unique images:", len(new_images))

print("\nMean Jaccard similarity:")
print(round(sum(overlap_scores) / len(overlap_scores), 4))

print("\n==========================================")