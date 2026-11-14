
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = PROJECT_ROOT / "results" / "knowledge_base" / "knowledge_base.jsonl"
TABLE_DIR = PROJECT_ROOT / "results" / "statistics" / "tables"
GRAPH_DIR = PROJECT_ROOT / "results" / "statistics" / "graphs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Count individual object categories
object_counts = Counter()
image_objects = []

with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        # Use a set so an object category is counted only once per image
        categories = {
            obj["category_name"]
            for obj in record["objects"]
        }

        image_objects.append(categories)
        object_counts.update(categories)

# Select the 20 most frequent categories
top_categories = [
    category
    for category, count in object_counts.most_common(20)
]

# Initialize co-occurrence matrix
matrix = pd.DataFrame(
    0,
    index=top_categories,
    columns=top_categories
)

# Count category pairs appearing in the same image
for categories in image_objects:
    selected = sorted(categories.intersection(top_categories))

    for category_a, category_b in combinations(selected, 2):
        matrix.loc[category_a, category_b] += 1
        matrix.loc[category_b, category_a] += 1

# Diagonal = number of images containing each category
for category in top_categories:
    matrix.loc[category, category] = sum(
        category in categories for categories in image_objects
    )

# Save matrix as CSV
matrix.to_csv(
    TABLE_DIR / "object_cooccurrence_matrix.csv"
)

# Draw heatmap
plt.figure(figsize=(14, 11))

plt.imshow(matrix.values, cmap="Blues", aspect="auto")
plt.colorbar(label="Number of images")

plt.xticks(
    range(len(top_categories)),
    top_categories,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(top_categories)),
    top_categories
)

plt.title("Object Category Co-occurrence Matrix")
plt.xlabel("Object Category")
plt.ylabel("Object Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "object_cooccurrence_matrix.png",
    dpi=300
)

plt.close()

print("\n========== OBJECT CO-OCCURRENCE ANALYSIS ==========")
print(f"Images analyzed: {len(image_objects)}")
print(f"Unique object categories: {len(object_counts)}")
print(f"Categories in matrix: {len(top_categories)}")

print("\nTop 20 categories:")
for category in top_categories:
    print(f"{category}: {matrix.loc[category, category]} images")

print("\nTop 10 object pairs:")

pair_counts = []

for category_a, category_b in combinations(top_categories, 2):
    pair_counts.append((
        category_a,
        category_b,
        matrix.loc[category_a, category_b]
    ))

pair_counts.sort(key=lambda item: item[2], reverse=True)

for category_a, category_b, count in pair_counts[:10]:
    print(f"{category_a} + {category_b}: {count} images")

print("\nCSV matrix and heatmap saved successfully.")
