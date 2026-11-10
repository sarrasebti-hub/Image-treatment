
import json
import csv
import os
from collections import Counter
from itertools import combinations

# ==========================================
# 1. CONFIGURATION
# ==========================================

METADATA_PATH = (
    "dataset/dataset/metadata/"
    "knowledge_base_metadata.json"
)

OUTPUT_DIR = "results/statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("======================================")
print(" OBJECT CO-OCCURRENCE ANALYSIS")
print("======================================")

# ==========================================
# 2. CHARGEMENT DES METADONNEES
# ==========================================

print("\n[1/4] Loading metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==========================================
# 3. CALCUL DES COOCCURRENCES
# ==========================================

print("\n[2/4] Calculating object co-occurrences...")

cooccurrence = Counter()

for item in data:

    categories = sorted(set(
        obj["category"]
        for obj in item.get("objects", [])
    ))

    for pair in combinations(categories, 2):
        cooccurrence[pair] += 1

# ==========================================
# 4. AFFICHAGE DES RESULTATS
# ==========================================

print("\n========== OBJECT CO-OCCURRENCE ==========")
print("Total images:", len(data))
print("Unique object pairs:", len(cooccurrence))

print("\nTOP 30 CO-OCCURRING OBJECT PAIRS:")

for rank, ((obj1, obj2), count) in enumerate(
    cooccurrence.most_common(30), start=1
):
    percentage = count / len(data) * 100

    print(
        f"{rank}. {obj1} + {obj2}: "
        f"{count} images ({percentage:.2f}%)"
    )

# ==========================================
# 5. EXPORT CSV
# ==========================================

output_path = os.path.join(
    OUTPUT_DIR,
    "object_cooccurrence.csv"
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Object A",
        "Object B",
        "Number of Images",
        "Percentage"
    ])

    for (obj1, obj2), count in cooccurrence.most_common():

        writer.writerow([
            obj1,
            obj2,
            count,
            round(count / len(data) * 100, 2)
        ])

# ==========================================
# 6. VALIDATION
# ==========================================

print("\n========== VALIDATION ==========")

assert len(data) == 10000
assert len(cooccurrence) == 1832

print("Images validated:", len(data))
print("Unique pairs validated:", len(cooccurrence))
print("CSV saved:", output_path)

print("\nOBJECT COOCCURRENCE ANALYSIS COMPLETED!")
