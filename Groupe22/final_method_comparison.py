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

old_map = {x["query_id"]: x for x in old_data}
new_map = {x["query_image_id"]: x for x in new_data}

common_ids = sorted(set(old_map) & set(new_map))

methods = {
    "Baseline FAISS": "baseline",
    "CLIP Reranking": "reranking",
    "MMR Lambda 0.7": "mmr"
}

scores = {name: [] for name in methods}
scores["MMR Lambda 0.9"] = []

for query_id in common_ids:

    old = old_map[query_id]
    new = new_map[query_id]

    for name, key in methods.items():

        for result in old[key]:
            if key == "baseline":
                scores[name].append(result["faiss_score"])
            elif key == "reranking":
                scores[name].append(result["clip_score"])
            else:
                scores[name].append(result["relevance_score"])

    for result in new["selected_captions"]:
        scores["MMR Lambda 0.9"].append(
            result["relevance_score"]
        )

print("\n========== FINAL METHOD COMPARISON ==========")
print("Common validation queries:", len(common_ids))

for method, values in scores.items():

    mean_score = sum(values) / len(values)

    print(f"\n{method}")
    print("Total results:", len(values))
    print("Mean score:", round(mean_score, 4))

print("\n=============================================")