
import os
import json
import re
import csv
from collections import Counter
import matplotlib.pyplot as plt

# ==========================================
# PARTIE 1 : CONFIGURATION
# ==========================================

annotations_path = "annotations/captions_train2017.json"
images_folder = "train2017"

os.makedirs("results/statistics", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

print("======================================")
print(" CAPTION DIVERSITY - COCO 2017")
print("======================================")

# ==========================================
# PARTIE 2 : DEFINIR LES TROIS GROUPES
# ==========================================

color_words = {
    "red", "blue", "green", "yellow", "black",
    "white", "brown", "orange", "pink", "purple",
    "gray", "grey"
}

attribute_words = {
    "large", "small", "big", "little", "young",
    "old", "tall", "short", "long", "short",
    "beautiful", "clean", "dirty", "wooden",
    "metal", "plastic", "round", "open", "closed",
    "full", "empty", "light", "dark"
}

context_words = {
    "street", "road", "park", "beach", "field",
    "kitchen", "room", "bedroom", "bathroom",
    "restaurant", "sidewalk", "grass", "mountain",
    "water", "sky", "snow", "forest", "building",
    "house", "office", "store", "court", "stadium",
    "outdoor", "indoor", "bridge", "airport"
}

categories = {
    "Colors": color_words,
    "Attributes": attribute_words,
    "Contexts": context_words
}

# ==========================================
# PARTIE 3 : CHARGER LES IMAGES LOCALES
# ==========================================

print("\n[1/5] Loading local image IDs...")

local_image_ids = set()

for filename in os.listdir(images_folder):
    if filename.lower().endswith(".jpg"):
        image_id = int(os.path.splitext(filename)[0])
        local_image_ids.add(image_id)

print("Local images:", len(local_image_ids))

# ==========================================
# PARTIE 4 : CHARGER LES CAPTIONS
# ==========================================

print("\n[2/5] Loading captions...")

with open(annotations_path, "r", encoding="utf-8") as f:
    data = json.load(f)

captions = []

for annotation in data["annotations"]:
    if annotation["image_id"] in local_image_ids:
        captions.append(annotation["caption"])

print("Total captions:", len(captions))

# ==========================================
# PARTIE 5 : ANALYSER LA DIVERSITE
# ==========================================

print("\n[3/5] Analyzing colors, attributes and contexts...")

term_frequency = {
    category: Counter()
    for category in categories
}

caption_coverage = Counter()

for caption in captions:

    words = re.findall(r"\b[a-zA-Z]+\b", caption.lower())
    word_set = set(words)

    for category, vocabulary in categories.items():

        # Nombre d'occurrences de chaque terme
        for word in words:
            if word in vocabulary:
                term_frequency[category][word] += 1

        # Nombre de captions contenant au moins un terme
        if word_set.intersection(vocabulary):
            caption_coverage[category] += 1

print("\n========== DIVERSITY RESULTS ==========")

for category in categories:

    print("\n", category)

    print("Total word occurrences:",
          sum(term_frequency[category].values()))

    print("Captions containing at least one term:",
          caption_coverage[category])

    percentage = caption_coverage[category] / len(captions) * 100

    print("Caption percentage:", round(percentage, 2), "%")

    print("Most frequent terms:")

    for word, count in term_frequency[category].most_common(10):
        print(f"  {word}: {count}")

print("\n======================================")

# ==========================================
# PARTIE 6 : SAUVEGARDER LES RESULTATS CSV
# ==========================================

print("\n[4/5] Saving CSV results...")

summary_path = "results/statistics/diversity_summary.csv"

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Category",
        "Total Word Occurrences",
        "Captions Containing Terms",
        "Percentage of Captions"
    ])

    for category in categories:

        total_occurrences = sum(term_frequency[category].values())
        number_of_captions = caption_coverage[category]
        percentage = number_of_captions / len(captions) * 100

        writer.writerow([
            category,
            total_occurrences,
            number_of_captions,
            round(percentage, 2)
        ])

details_path = "results/statistics/diversity_terms.csv"

with open(details_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Category",
        "Term",
        "Frequency",
        "Number of Captions"
    ])

    for category in categories:

        for word, frequency in term_frequency[category].most_common():

            number_of_captions = sum(
                1 for caption in captions
                if word in re.findall(r"\b[a-zA-Z]+\b", caption.lower())
            )

            writer.writerow([
                category,
                word,
                frequency,
                number_of_captions
            ])

print("CSV files saved successfully!")

# ==========================================
# PARTIE 7 : CREER LE GRAPHIQUE
# ==========================================

print("\n[5/5] Creating diversity chart...")

category_names = list(categories.keys())

category_values = [
    caption_coverage[category]
    for category in category_names
]

plt.figure(figsize=(9, 6))

plt.bar(category_names, category_values)

plt.xlabel("Diversity Categories")
plt.ylabel("Number of Captions")
plt.title("Caption Diversity - COCO 2017")

plt.tight_layout()

chart_path = "results/visualizations/caption_diversity.png"

plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nChart saved:", chart_path)
print("\nCaption diversity analysis completed!")
