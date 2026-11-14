
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = PROJECT_ROOT / "results" / "knowledge_base" / "knowledge_base.jsonl"
TABLE_DIR = PROJECT_ROOT / "results" / "statistics" / "tables"
GRAPH_DIR = PROJECT_ROOT / "results" / "statistics" / "graphs"

# Create output folders
TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Count objects in each image
image_object_counts = []

with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        image_id = record["image_id"]
        object_count = len(record["objects"])

        image_object_counts.append({
            "Image ID": image_id,
            "Object Count": object_count
        })

# Create DataFrame
df = pd.DataFrame(image_object_counts)

# Calculate statistics
total_images = len(df)
average_objects = df["Object Count"].mean()
median_objects = df["Object Count"].median()
minimum_objects = df["Object Count"].min()
maximum_objects = df["Object Count"].max()
std_objects = df["Object Count"].std()

# Group images by complexity
def classify_scene(count):
    if count <= 2:
        return "Simple"
    elif count <= 4:
        return "Medium"
    elif count <= 10:
        return "Complex"
    else:
        return "Very Complex"

df["Complexity"] = df["Object Count"].apply(classify_scene)

complexity_distribution = (
    df["Complexity"]
    .value_counts()
    .reindex(["Simple", "Medium", "Complex", "Very Complex"], fill_value=0)
)

# Save detailed table
df.to_csv(
    TABLE_DIR / "scene_complexity_per_image.csv",
    index=False
)

# Save summary table
summary = pd.DataFrame({
    "Metric": [
        "Total Images",
        "Average Objects per Image",
        "Median Objects per Image",
        "Minimum Objects",
        "Maximum Objects",
        "Standard Deviation"
    ],
    "Value": [
        total_images,
        round(average_objects, 2),
        median_objects,
        minimum_objects,
        maximum_objects,
        round(std_objects, 2)
    ]
})

summary.to_csv(
    TABLE_DIR / "scene_complexity_summary.csv",
    index=False
)

# Save complexity distribution
distribution = complexity_distribution.reset_index()
distribution.columns = ["Complexity", "Number of Images"]

distribution["Percentage"] = (
    distribution["Number of Images"] / total_images * 100
).round(2)

distribution.to_csv(
    TABLE_DIR / "scene_complexity_distribution.csv",
    index=False
)

# Generate histogram
plt.figure(figsize=(10, 6))

plt.hist(
    df["Object Count"],
    bins=range(0, int(maximum_objects) + 2),
    edgecolor="black"
)

plt.xlabel("Number of Objects per Image")
plt.ylabel("Number of Images")
plt.title("Distribution of Objects per Image")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "scene_complexity_histogram.png",
    dpi=300
)

plt.close()

# Generate complexity distribution chart
plt.figure(figsize=(9, 6))

plt.bar(
    distribution["Complexity"],
    distribution["Number of Images"]
)

plt.xlabel("Scene Complexity")
plt.ylabel("Number of Images")
plt.title("Distribution of Scene Complexity")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "scene_complexity_distribution.png",
    dpi=300
)

plt.close()

# Display results
print("========== SCENE COMPLEXITY ANALYSIS ==========")
print(f"Total images: {total_images}")
print(f"Average objects per image: {average_objects:.2f}")
print(f"Median objects per image: {median_objects}")
print(f"Minimum objects: {minimum_objects}")
print(f"Maximum objects: {maximum_objects}")
print(f"Standard deviation: {std_objects:.2f}")

print("\n========== COMPLEXITY DISTRIBUTION ==========")
print(distribution.to_string(index=False))

print("\nAll tables and graphs saved successfully.")
