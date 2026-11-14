
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = PROJECT_ROOT / "results" / "knowledge_base" / "knowledge_base.jsonl"
TABLE_DIR = PROJECT_ROOT / "results" / "statistics" / "tables"
GRAPH_DIR = PROJECT_ROOT / "results" / "statistics" / "graphs"

# Create output folders if they do not exist
TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Count object categories
object_counts = Counter()

with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        for obj in record["objects"]:
            category = obj["category_name"]
            object_counts[category] += 1

# Calculate total annotations
total_objects = sum(object_counts.values())

# Create a table with the 50 most frequent categories
top_50 = object_counts.most_common(50)

df = pd.DataFrame(top_50, columns=["Object Category", "Frequency"])

df["Percentage"] = (
    df["Frequency"] / total_objects * 100
).round(2)

# Save the CSV table
table_path = TABLE_DIR / "top_50_objects.csv"
df.to_csv(table_path, index=False)

# Generate bar chart
plt.figure(figsize=(16, 10))

plt.barh(
    df["Object Category"],
    df["Frequency"]
)

plt.xlabel("Number of Object Annotations")
plt.ylabel("Object Category")
plt.title("Top 50 Most Frequent Object Categories in the Knowledge Base")

plt.gca().invert_yaxis()
plt.tight_layout()

graph_path = GRAPH_DIR / "top_50_objects.png"
plt.savefig(graph_path, dpi=300)
plt.close()

# Display results
print("========== TOP 50 OBJECT CATEGORIES ==========")
print(f"Total images analyzed: 10,000")
print(f"Total object annotations: {total_objects}")
print(f"Unique object categories: {len(object_counts)}")
print()
print(df.to_string(index=False))
print()
print("CSV saved to:", table_path)
print("Graph saved to:", graph_path)
print("Analysis completed successfully.")
