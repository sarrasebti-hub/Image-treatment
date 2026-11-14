
import json
from collections import Counter
from itertools import permutations
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


def get_spatial_relationship(obj1, obj2):
    """
    Infer the spatial relationship between two objects
    using their COCO bounding boxes.

    Bounding box format: [x, y, width, height]
    """

    x1, y1, w1, h1 = obj1["bbox"]
    x2, y2, w2, h2 = obj2["bbox"]

    # Calculate object centers
    center_x1 = x1 + w1 / 2
    center_y1 = y1 + h1 / 2

    center_x2 = x2 + w2 / 2
    center_y2 = y2 + h2 / 2

    # Calculate intersection
    intersection_left = max(x1, x2)
    intersection_top = max(y1, y2)
    intersection_right = min(x1 + w1, x2 + w2)
    intersection_bottom = min(y1 + h1, y2 + h2)

    intersection_width = max(0, intersection_right - intersection_left)
    intersection_height = max(0, intersection_bottom - intersection_top)

    intersection_area = intersection_width * intersection_height

    area1 = w1 * h1
    area2 = w2 * h2

    # Check whether objects significantly overlap
    smaller_area = min(area1, area2)

    if smaller_area > 0:
        overlap_ratio = intersection_area / smaller_area
    else:
        overlap_ratio = 0

    if overlap_ratio >= 0.10:
        return "overlapping"

    # Determine the dominant spatial direction
    dx = center_x2 - center_x1
    dy = center_y2 - center_y1

    if abs(dx) >= abs(dy):
        return "left_of" if dx > 0 else "right_of"

    return "above" if dy > 0 else "below"


# Count spatial relationships
relationship_counts = Counter()
total_images = 0
images_with_pairs = 0
total_pairs = 0

print("Analyzing spatial relationships...")

with open(KB_FILE, "r", encoding="utf-8") as file:

    for line in file:
        record = json.loads(line)

        total_images += 1
        objects = record["objects"]

        # Images need at least two objects
        if len(objects) < 2:
            continue

        images_with_pairs += 1

        # Analyze every ordered pair of different objects
        for obj1, obj2 in permutations(objects, 2):

            relationship = get_spatial_relationship(obj1, obj2)

            relationship_counts[relationship] += 1
            total_pairs += 1


# Create results table
df = pd.DataFrame(
    relationship_counts.most_common(),
    columns=["Spatial Relationship", "Frequency"]
)

df["Percentage"] = (
    df["Frequency"] / total_pairs * 100
).round(2)

# Save CSV table
df.to_csv(
    TABLE_DIR / "spatial_relationship_distribution.csv",
    index=False
)

# Generate bar chart
plt.figure(figsize=(10, 6))

plt.bar(
    df["Spatial Relationship"],
    df["Frequency"]
)

plt.xlabel("Spatial Relationship")
plt.ylabel("Number of Directed Object Pairs")
plt.title("Distribution of Inferred Spatial Relationships")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "spatial_relationship_distribution.png",
    dpi=300
)

plt.close()

# Display results
print("\n========== SPATIAL RELATIONSHIP ANALYSIS ==========")
print(f"Total images analyzed: {total_images}")
print(f"Images with at least two objects: {images_with_pairs}")
print(f"Total directed object pairs: {total_pairs}")
print()

print(df.to_string(index=False))

print("\nCSV table and graph saved successfully.")
print("Analysis completed.")
