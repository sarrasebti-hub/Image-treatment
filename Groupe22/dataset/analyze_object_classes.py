
import os
import json
import csv
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# ==========================================
# PARTIE 1 : CONFIGURATION
# ==========================================

annotations_path = "annotations/instances_train2017.json"
images_folder = "train2017"

os.makedirs("results/statistics", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

print("======================================")
print(" TOP 50 OBJECT CLASSES - COCO 2017")
print("======================================")

# ==========================================
# PARTIE 2 : CHARGER LES IMAGES LOCALES
# ==========================================

print("\n[1/5] Loading local image IDs...")

local_image_ids = set()

for filename in os.listdir(images_folder):
    if filename.lower().endswith(".jpg"):
        image_id = int(os.path.splitext(filename)[0])
        local_image_ids.add(image_id)

print("Number of local images:", len(local_image_ids))

# ==========================================
# PARTIE 3 : CHARGER LES ANNOTATIONS COCO
# ==========================================

print("\n[2/5] Loading COCO annotations...")

with open(annotations_path, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

print("Number of COCO categories:", len(categories))

# ==========================================
# PARTIE 4 : COMPTER LES OBJETS
# ==========================================

print("\n[3/5] Counting object classes...")

object_counts = Counter()
image_counts = defaultdict(set)

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    # Garder uniquement les images présentes localement
    if image_id not in local_image_ids:
        continue

    category_id = annotation["category_id"]
    category_name = categories[category_id]

    # Compter les objets
    object_counts[category_name] += 1

    # Compter les images contenant cette classe
    image_counts[category_name].add(image_id)

# ==========================================
# PARTIE 5 : AFFICHER LE TOP 50
# ==========================================

print("\n[4/5] Selecting the top 50 classes...")

top_50 = object_counts.most_common(50)

print("\n========== TOP 50 OBJECT CLASSES ==========")

for rank, (name, count) in enumerate(top_50, start=1):

    number_of_images = len(image_counts[name])

    print(
        f"{rank}. {name}: "
        f"{count} objects, "
        f"{number_of_images} images"
    )

print("===========================================")

print("\nTotal object instances:", sum(object_counts.values()))
print("Number of classes found:", len(object_counts))

# ==========================================
# PARTIE 6 : SAUVEGARDER LE TABLEAU CSV
# ==========================================

csv_path = "results/statistics/top_50_object_classes.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Rank",
        "Object Class",
        "Number of Objects",
        "Number of Images"
    ])

    for rank, (name, count) in enumerate(top_50, start=1):

        writer.writerow([
            rank,
            name,
            count,
            len(image_counts[name])
        ])

print("\nCSV file saved:", csv_path)

# ==========================================
# PARTIE 7 : CREER LE GRAPHIQUE
# ==========================================

names = [item[0] for item in top_50]
counts = [item[1] for item in top_50]

plt.figure(figsize=(14, 18))

plt.barh(names[::-1], counts[::-1])

plt.xlabel("Number of Object Instances")
plt.ylabel("Object Classes")

plt.title("Top 50 Object Classes - COCO 2017")

plt.tight_layout()

chart_path = "results/visualizations/top_50_object_classes.png"

plt.savefig(chart_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nChart saved:", chart_path)

# ==========================================
# FIN
# ==========================================

print("\nAnalysis completed successfully!")
