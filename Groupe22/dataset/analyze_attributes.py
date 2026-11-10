
import json
import re
import csv
import os
from collections import defaultdict

# ==========================================
# 1. CONFIGURATION
# ==========================================

METADATA_PATH = (
    "dataset/dataset/metadata/"
    "knowledge_base_metadata.json"
)

OUTPUT_DIR = "results/statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

groups = {
    "Colors": [
        "white", "black", "red", "blue", "green",
        "yellow", "orange", "pink", "purple",
        "brown", "gray", "grey"
    ],
    "Shapes": [
        "round", "rounded", "square", "rectangular",
        "rectangle", "circular", "oval", "triangular"
    ],
    "Contexts": [
        "indoor", "outdoor", "kitchen", "bedroom",
        "bathroom", "office", "street", "road",
        "beach", "park", "field", "restaurant"
    ],
    "Lighting_Time": [
        "day", "daytime", "night", "nighttime",
        "dark", "bright", "sunny", "sunlight"
    ]
}

# Gray and Grey are combined in the final report
aliases = {
    "gray": "gray/grey",
    "grey": "gray/grey"
}

# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

print("==========================================")
print(" ATTRIBUTE AND CONTEXT ANALYSIS")
print("==========================================")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Images:", len(data))

# ==========================================
# 3. PREPARATION DES COMPTEURS
# ==========================================

results = defaultdict(
    lambda: {"captions": 0, "images": set()}
)

print("\nAnalyzing captions...")

total_captions = 0

# ==========================================
# 4. EXTRACTION DES TERMES
# ==========================================

for image in data:
    image_id = image["image_id"]

    for caption in image["captions"]:
        total_captions += 1

        tokens = set(
            re.findall(r"\b[a-z]+\b", caption.lower())
        )

        for group, words in groups.items():

            for word in words:
                if word in tokens:

                    key = aliases.get(word, word)
                    result_key = (group, key)

                    results[result_key]["captions"] += 1
                    results[result_key]["images"].add(image_id)

# ==========================================
# 5. AFFICHAGE DES RESULTATS
# ==========================================

print("\n========== RESULTS ==========")

for group in groups:

    print(f"\n{group}")

    group_results = [
        (word, values)
        for (category, word), values in results.items()
        if category == group
    ]

    for word, values in sorted(group_results):
        print(
            f"{word}: "
            f"captions={values['captions']}, "
            f"images={len(values['images'])}"
        )

# ==========================================
# 6. SAUVEGARDE CSV
# ==========================================

output_path = os.path.join(
    OUTPUT_DIR,
    "attribute_context_statistics.csv"
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Group",
        "Term",
        "Caption Count",
        "Image Count"
    ])

    for (group, word), values in sorted(results.items()):

        writer.writerow([
            group,
            word,
            values["captions"],
            len(values["images"])
        ])

print("\n========== VALIDATION ==========")
print("Total images:", len(data))
print("Total captions:", total_captions)
print("CSV saved:", output_path)

assert len(data) == 10000
assert total_captions == 50000

print("ATTRIBUTE ANALYSIS COMPLETED!")
