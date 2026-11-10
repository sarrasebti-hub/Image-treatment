
import json
import csv
import os
from collections import Counter

# ==========================================
# 1. CONFIGURATION
# ==========================================

METADATA_PATH = (
    "dataset/dataset/metadata/"
    "knowledge_base_metadata.json"
)

OUTPUT_DIR = "results/statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. FONCTIONS GEOMETRIQUES
# ==========================================

def get_relation(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Centers
    acx = ax + aw / 2
    acy = ay + ah / 2

    bcx = bx + bw / 2
    bcy = by + bh / 2

    # Actual rectangle intersection
    intersection_width = max(
        0, min(ax + aw, bx + bw) - max(ax, bx)
    )

    intersection_height = max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )

    if intersection_width > 0 and intersection_height > 0:
        return "overlapping"

    # Direction based on center positions
    dx = bcx - acx
    dy = bcy - acy

    if abs(dx) >= abs(dy):
        return "left_of" if dx > 0 else "right_of"

    return "above" if dy > 0 else "below"


# ==========================================
# 3. CHARGEMENT DES DONNEES
# ==========================================

print("======================================")
print(" SPATIAL RELATIONS ANALYSIS")
print("======================================")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==========================================
# 4. ANALYSE DES RELATIONS
# ==========================================

relations = Counter()
images_with_relations = Counter()
total_pairs = 0

pair_records = []

for item in data:
    objects = item.get("objects", [])
    image_id = item["image_id"]

    image_relations = set()

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):

            a = objects[i]
            b = objects[j]

            relation = get_relation(a["bbox"], b["bbox"])

            relations[relation] += 1
            image_relations.add(relation)
            total_pairs += 1

            pair_records.append([
                image_id,
                a["category"],
                b["category"],
                relation
            ])

    for relation in image_relations:
        images_with_relations[relation] += 1

# ==========================================
# 5. AFFICHAGE DES RESULTATS
# ==========================================

print("\n========== SPATIAL RELATIONS ==========")
print("Total images:", len(data))
print("Total object pairs:", total_pairs)

for relation, count in relations.most_common():

    percentage = count / total_pairs * 100 if total_pairs else 0

    print(
        f"{relation}: {count} pairs "
        f"({percentage:.2f}%), "
        f"{images_with_relations[relation]} images"
    )

# ==========================================
# 6. SAUVEGARDE CSV
# ==========================================

output_path = os.path.join(
    OUTPUT_DIR,
    "spatial_relations.csv"
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Image ID",
        "Object A",
        "Object B",
        "Estimated Relation"
    ])

    writer.writerows(pair_records)

print("\nCSV saved:", output_path)

# ==========================================
# 7. VALIDATION
# ==========================================

expected_pairs = sum(
    len(item["objects"]) * (len(item["objects"]) - 1) // 2
    for item in data
)

assert total_pairs == expected_pairs
assert sum(relations.values()) == total_pairs

print("\nTotal pairs validated:", total_pairs)
print("Spatial relations analysis completed!")
