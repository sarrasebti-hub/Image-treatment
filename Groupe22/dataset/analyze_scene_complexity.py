
import json
import csv
import os
from statistics import mean, median
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

METADATA_PATH = (
    "dataset/dataset/metadata/"
    "knowledge_base_metadata.json"
)

os.makedirs("results/statistics", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

print("======================================")
print(" SCENE COMPLEXITY - KNOWLEDGE BASE")
print("======================================")

# ==========================================
# 2. CHARGEMENT DES METADONNEES
# ==========================================

print("\n[1/5] Loading Knowledge Base metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==========================================
# 3. COMPTER LES OBJETS PAR IMAGE
# ==========================================

print("\n[2/5] Counting objects per image...")

object_per_image = {}

for image in data:
    image_id = image["image_id"]
    number_objects = len(image["objects"])

    object_per_image[image_id] = number_objects

counts = list(object_per_image.values())

if not counts:
    raise ValueError("No images found in metadata.")

# ==========================================
# 4. CALCULER LES STATISTIQUES
# ==========================================

print("\n[3/5] Calculating scene complexity...")

average_objects = mean(counts)
median_objects = median(counts)
minimum_objects = min(counts)
maximum_objects = max(counts)

print("\n========== SCENE COMPLEXITY ==========")
print("Total images:", len(counts))
print("Total objects:", sum(counts))
print("Average objects per image:", round(average_objects, 3))
print("Median objects per image:", median_objects)
print("Minimum objects:", minimum_objects)
print("Maximum objects:", maximum_objects)

# Complexity groups
complexity_groups = {
    "0-2 objects": sum(0 <= n <= 2 for n in counts),
    "3-5 objects": sum(3 <= n <= 5 for n in counts),
    "6-10 objects": sum(6 <= n <= 10 for n in counts),
    "11-20 objects": sum(11 <= n <= 20 for n in counts),
    "More than 20 objects": sum(n > 20 for n in counts)
}

print("\nComplexity distribution:")

for group, number in complexity_groups.items():
    percentage = number / len(counts) * 100
    print(f"{group}: {number} images ({percentage:.2f}%)")

print("======================================")

# ==========================================
# 5. SAUVEGARDER LES RESULTATS CSV
# ==========================================

details_path = (
    "results/statistics/"
    "scene_complexity_per_image.csv"
)

with open(details_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow(["Image ID", "Number of Objects"])

    for image_id, number in sorted(object_per_image.items()):
        writer.writerow([image_id, number])

summary_path = (
    "results/statistics/"
    "scene_complexity_summary.csv"
)

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow(["Metric", "Value"])
    writer.writerow(["Total images", len(counts)])
    writer.writerow(["Total objects", sum(counts)])
    writer.writerow(["Average objects per image", average_objects])
    writer.writerow(["Median objects per image", median_objects])
    writer.writerow(["Minimum objects", minimum_objects])
    writer.writerow(["Maximum objects", maximum_objects])

    writer.writerow([])
    writer.writerow(["Complexity Group", "Number of Images", "Percentage"])

    for group, number in complexity_groups.items():
        percentage = number / len(counts) * 100
        writer.writerow([group, number, round(percentage, 2)])

print("\nCSV files saved successfully!")

# ==========================================
# 6. CREER LE GRAPHIQUE
# ==========================================

print("\n[4/5] Generating chart...")

group_names = list(complexity_groups.keys())
group_values = list(complexity_groups.values())

plt.figure(figsize=(11, 6))
plt.bar(group_names, group_values)

plt.xlabel("Number of Objects per Image")
plt.ylabel("Number of Images")
plt.title("Scene Complexity Distribution - Knowledge Base")

plt.xticks(rotation=20)
plt.tight_layout()

chart_path = (
    "results/visualizations/"
    "scene_complexity.png"
)

plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print("Chart saved:", chart_path)

# ==========================================
# 7. VALIDATION
# ==========================================

print("\n[5/5] Validating results...")

assert len(counts) == 10000, "Unexpected image count!"
assert sum(counts) == 72754, "Unexpected object count!"
assert sum(complexity_groups.values()) == len(counts)

print("Image count validated.")
print("Object count validated.")
print("Complexity distribution validated.")

print("\nSCENE COMPLEXITY ANALYSIS COMPLETED!")
