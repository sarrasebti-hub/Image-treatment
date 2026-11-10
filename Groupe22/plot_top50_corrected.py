
import csv
import os
import matplotlib.pyplot as plt

CSV_PATH = "results/statistics/top_50_object_classes_corrected.csv"
OUTPUT_DIR = "results/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

names = []
counts = []

with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        names.append(row["Object Class"])
        counts.append(int(row["Number of Objects"]))

assert len(names) == 50
assert len(counts) == 50

plt.figure(figsize=(14, 18))
plt.barh(names[::-1], counts[::-1])

plt.xlabel("Number of Object Instances")
plt.ylabel("Object Classes")
plt.title("Top 50 Object Classes - Knowledge Base (10,000 Images)")

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "top_50_object_classes_corrected.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Number of classes:", len(names))
print("Graph saved:", output_path)
print("Validation passed.")
