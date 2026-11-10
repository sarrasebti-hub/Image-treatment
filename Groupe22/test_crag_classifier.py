
import json
from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    BASE_DIR
    / "dataset"
    / "embeddings"
    / "crag_calibration_results.json"
)

# Seuils expérimentaux issus de P25 et P75
LOWER_THRESHOLD = 0.3170
UPPER_THRESHOLD = 0.3402


# =====================================================
# 2. CLASSIFICATION CRAG
# =====================================================

def classify_relevance(score):

    if score >= UPPER_THRESHOLD:
        return "CORRECT"

    elif score >= LOWER_THRESHOLD:
        return "AMBIGUOUS"

    else:
        return "INCORRECT"


# =====================================================
# 3. CHARGEMENT DES RESULTATS
# =====================================================

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)


# =====================================================
# 4. CLASSIFICATION DES IMAGES
# =====================================================

counts = {
    "CORRECT": 0,
    "AMBIGUOUS": 0,
    "INCORRECT": 0
}

print("\n========================================")
print("          CRAG CLASSIFICATION")
print("========================================")

for item in results:

    score = item["best_clip_score"]

    category = classify_relevance(score)

    counts[category] += 1

    print(
        f"Image: {item['query_id']} | "
        f"Score: {score:.4f} | "
        f"Decision: {category}"
    )


# =====================================================
# 5. STATISTIQUES
# =====================================================

print("\n========================================")
print("          CLASSIFICATION SUMMARY")
print("========================================")

total = len(results)

for category, count in counts.items():

    percentage = (
        count / total * 100
        if total > 0
        else 0
    )

    print(
        f"{category}: {count}/{total} "
        f"({percentage:.1f}%)"
    )

print("\nNombre total :", total)

print("\nSeuil inferieur :", LOWER_THRESHOLD)
print("Seuil superieur :", UPPER_THRESHOLD)

print("\nClassification terminee.")
