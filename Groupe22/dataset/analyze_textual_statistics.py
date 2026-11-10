
import os
import json
import re
import csv
from collections import Counter
from statistics import mean, median
import matplotlib.pyplot as plt
import nltk

# ==========================================
# PARTIE 1 : CONFIGURATION
# ==========================================

annotations_path = "annotations/captions_train2017.json"
images_folder = "train2017"

os.makedirs("results/statistics", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

print("======================================")
print(" TEXTUAL STATISTICS - COCO 2017")
print("======================================")

# ==========================================
# PARTIE 2 : PREPARER NLTK (POS)
# ==========================================

print("\n[1/6] Preparing NLTK...")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

# ==========================================
# PARTIE 3 : CHARGER LES IMAGES LOCALES
# ==========================================

print("\n[2/6] Loading local image IDs...")

local_image_ids = set()

for filename in os.listdir(images_folder):
    if filename.lower().endswith(".jpg"):
        image_id = int(os.path.splitext(filename)[0])
        local_image_ids.add(image_id)

print("Local images:", len(local_image_ids))

# ==========================================
# PARTIE 4 : CHARGER LES CAPTIONS
# ==========================================

print("\n[3/6] Loading captions...")

with open(annotations_path, "r", encoding="utf-8") as f:
    data = json.load(f)

captions = []

for annotation in data["annotations"]:
    if annotation["image_id"] in local_image_ids:
        captions.append(annotation["caption"])

print("Total captions analyzed:", len(captions))

# ==========================================
# PARTIE 5 : STATISTIQUES TEXTUELLES
# ==========================================

print("\n[4/6] Analyzing caption statistics...")

caption_lengths = []
all_words = []
word_frequency = Counter()

# Mots anglais courants à exclure du classement
stop_words = {
    "a", "an", "the", "is", "are", "was", "were",
    "in", "on", "at", "of", "to", "and", "with",
    "for", "by", "from", "as", "this", "that",
    "it", "his", "her", "their", "he", "she"
}

for caption in captions:

    words = re.findall(r"\b[a-zA-Z]+\b", caption.lower())

    caption_lengths.append(len(words))
    all_words.extend(words)

    for word in words:
        if word not in stop_words:
            word_frequency[word] += 1

total_words = len(all_words)
vocabulary_size = len(set(all_words))

average_length = mean(caption_lengths)
median_length = median(caption_lengths)
minimum_length = min(caption_lengths)
maximum_length = max(caption_lengths)

print("\n========== TEXTUAL STATISTICS ==========")

print("Total captions:", len(captions))
print("Total words:", total_words)
print("Vocabulary size:", vocabulary_size)

print("Average caption length:", round(average_length, 2))
print("Median caption length:", median_length)
print("Minimum caption length:", minimum_length)
print("Maximum caption length:", maximum_length)

print("\nTop 20 most frequent words:")

top_words = word_frequency.most_common(20)

for rank, (word, count) in enumerate(top_words, start=1):
    print(f"{rank}. {word}: {count}")

print("========================================")

# ==========================================
# PARTIE 6 : ANALYSE GRAMMATICALE (POS)
# ==========================================

print("\n[5/6] Performing POS analysis...")

pos_counts = Counter()

for caption in captions:

    words = re.findall(r"\b[a-zA-Z]+\b", caption.lower())

    if not words:
        continue

    tagged_words = nltk.pos_tag(words)

    for word, tag in tagged_words:

        if tag.startswith("NN"):
            pos_counts["Nouns"] += 1

        elif tag.startswith("VB"):
            pos_counts["Verbs"] += 1

        elif tag.startswith("JJ"):
            pos_counts["Adjectives"] += 1

        elif tag.startswith("RB"):
            pos_counts["Adverbs"] += 1

print("\nPOS Distribution:")

for category, count in pos_counts.most_common():
    print(category, ":", count)

# ==========================================
# PARTIE 7 : SAUVEGARDER LES STATISTIQUES
# ==========================================

summary_path = "results/statistics/textual_statistics_summary.csv"

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow(["Metric", "Value"])

    writer.writerow(["Total captions", len(captions)])
    writer.writerow(["Total words", total_words])
    writer.writerow(["Vocabulary size", vocabulary_size])
    writer.writerow(["Average caption length", round(average_length, 2)])
    writer.writerow(["Median caption length", median_length])
    writer.writerow(["Minimum caption length", minimum_length])
    writer.writerow(["Maximum caption length", maximum_length])

    writer.writerow([])
    writer.writerow(["POS Category", "Count"])

    for category, count in pos_counts.items():
        writer.writerow([category, count])

# Sauvegarder les mots fréquents
words_path = "results/statistics/most_frequent_words.csv"

with open(words_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["Word", "Frequency"])

    for word, count in top_words:
        writer.writerow([word, count])

# Sauvegarder les longueurs des captions
lengths_path = "results/statistics/caption_lengths.csv"

with open(lengths_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["Caption ID", "Number of Words"])

    for index, length in enumerate(caption_lengths, start=1):
        writer.writerow([index, length])

print("\nCSV files saved successfully!")

# ==========================================
# PARTIE 8 : GRAPHIQUE DES LONGUEURS
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(caption_lengths, bins=20, edgecolor="black")

plt.xlabel("Number of Words")
plt.ylabel("Number of Captions")
plt.title("Caption Length Distribution - COCO 2017")

plt.tight_layout()

plt.savefig(
    "results/visualizations/caption_length_distribution.png",
    dpi=300
)

plt.show()

# ==========================================
# PARTIE 9 : GRAPHIQUE DES MOTS FREQUENTS
# ==========================================

plt.figure(figsize=(10, 7))

words = [item[0] for item in top_words]
frequencies = [item[1] for item in top_words]

plt.barh(words[::-1], frequencies[::-1])

plt.xlabel("Frequency")
plt.ylabel("Words")
plt.title("Top 20 Most Frequent Words - COCO 2017")

plt.tight_layout()

plt.savefig(
    "results/visualizations/top_20_frequent_words.png",
    dpi=300
)

plt.show()

# ==========================================
# PARTIE 10 : GRAPHIQUE POS
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(pos_counts.keys(), pos_counts.values())

plt.xlabel("Grammatical Category")
plt.ylabel("Number of Words")
plt.title("POS Distribution - COCO 2017")

plt.tight_layout()

plt.savefig(
    "results/visualizations/pos_distribution.png",
    dpi=300
)

plt.show()

print("\nAll textual statistics analyses completed!")
