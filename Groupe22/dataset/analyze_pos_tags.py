import json
import csv
import os
from collections import Counter

import nltk
from nltk import word_tokenize, pos_tag

# ==========================================
# 1. CONFIGURATION
# ==========================================

METADATA_PATH = (
    "dataset/dataset/metadata/"
    "knowledge_base_metadata.json"
)

OUTPUT_DIR = "results/statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("======================================")
print(" POS TAGGING - KNOWLEDGE BASE")
print("======================================")

# ==========================================
# 2. CHARGEMENT DES CAPTIONS
# ==========================================

print("\n[1/5] Loading captions...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

captions = [
    caption
    for image in data
    for caption in image["captions"]
]

print("Total images:", len(data))
print("Total captions:", len(captions))

# ==========================================
# 3. POS TAGGING PAR CAPTION
# ==========================================

print("\n[2/5] Performing POS tagging...")

tag_counts = Counter()
total_tokens = 0

for index, caption in enumerate(captions, start=1):

    tokens = word_tokenize(caption)
    tagged_words = pos_tag(tokens)

    for word, tag in tagged_words:
        tag_counts[tag] += 1
        total_tokens += 1

    if index % 5000 == 0:
        print(f"Processed {index}/{len(captions)} captions")

# ==========================================
# 4. REGROUPEMENT DES CATEGORIES
# ==========================================

print("\n[3/5] Grouping grammatical categories...")

categories = {
    "Nouns": ["NN", "NNS", "NNP", "NNPS"],
    "Verbs": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "Adjectives": ["JJ", "JJR", "JJS"],
    "Adverbs": ["RB", "RBR", "RBS"],
    "Determiners": ["DT", "PDT", "WDT"],
    "Prepositions": ["IN"],
    "Pronouns": ["PRP", "PRP$", "WP", "WP$"]
}

category_counts = {}

for category, tags in categories.items():
    category_counts[category] = sum(
        tag_counts[tag] for tag in tags
    )

# ==========================================
# 5. AFFICHAGE DES RESULTATS
# ==========================================

print("\n========== POS TAGGING RESULTS ==========")

print("Total tagged tokens:", total_tokens)

for category, count in category_counts.items():
    percentage = count / total_tokens * 100

    print(
        f"{category}: {count} "
        f"({percentage:.2f}%)"
    )

print("\n========== TOP 15 POS TAGS ==========")

for tag, count in tag_counts.most_common(15):
    print(f"{tag}: {count}")

# ==========================================
# 6. SAUVEGARDE CSV
# ==========================================

print("\n[4/5] Saving results...")

output_path = os.path.join(
    OUTPUT_DIR,
    "pos_tagging_statistics.csv"
)

with open(output_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow(["Category", "Count", "Percentage"])

    for category, count in category_counts.items():
        percentage = count / total_tokens * 100

        writer.writerow([
            category,
            count,
            round(percentage, 2)
        ])

    writer.writerow([])
    writer.writerow(["POS Tag", "Frequency"])

    for tag, count in tag_counts.most_common():
        writer.writerow([tag, count])

print("CSV saved:", output_path)

# ==========================================
# 7. VALIDATION
# ==========================================

print("\n[5/5] Validating results...")

assert len(data) == 10000
assert len(captions) == 50000
assert sum(tag_counts.values()) == total_tokens

print("Images validated: 10000")
print("Captions validated: 50000")
print("POS tagging completed successfully!")