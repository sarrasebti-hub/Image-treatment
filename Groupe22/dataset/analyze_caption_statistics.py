import json
import re
import csv
import os
from collections import Counter
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
print(" CAPTION STATISTICS - KNOWLEDGE BASE")
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

if not captions:
    raise ValueError("No captions found.")

print("Total images:", len(data))
print("Total captions:", len(captions))

# ==========================================
# 3. TOKENISATION
# ==========================================

print("\n[2/5] Tokenizing captions...")

tokenized_captions = [
    re.findall(r"\b[a-z0-9]+\b", caption.lower())
    for caption in captions
]

lengths = [len(tokens) for tokens in tokenized_captions]

all_words = [
    word
    for tokens in tokenized_captions
    for word in tokens
]

vocabulary = set(all_words)
word_counts = Counter(all_words)

# ==========================================
# 4. STATISTIQUES
# ==========================================

print("\n[3/5] Calculating statistics...")

total_words = len(all_words)
average_length = mean(lengths)
median_length = median(lengths)

print("\n========== CAPTION STATISTICS ==========")
print("Total captions:", len(captions))
print("Total word tokens:", total_words)
print("Vocabulary size:", len(vocabulary))
print("Average words per caption:", round(average_length, 3))
print("Median words per caption:", median_length)
print("Minimum caption length:", min(lengths))
print("Maximum caption length:", max(lengths))

print("\n========== TOP 20 WORDS ==========")

for rank, (word, count) in enumerate(
    word_counts.most_common(20), start=1
):
    print(f"{rank}. {word}: {count}")

# ==========================================
# 5. DISTRIBUTION DES LONGUEURS
# ==========================================

length_groups = {
    "6-8 words": sum(6 <= n <= 8 for n in lengths),
    "9-11 words": sum(9 <= n <= 11 for n in lengths),
    "12-15 words": sum(12 <= n <= 15 for n in lengths),
    "16-20 words": sum(16 <= n <= 20 for n in lengths),
    "More than 20 words": sum(n > 20 for n in lengths)
}

print("\n========== CAPTION LENGTH DISTRIBUTION ==========")

for group, number in length_groups.items():
    percentage = number / len(captions) * 100
    print(f"{group}: {number} ({percentage:.2f}%)")

# ==========================================
# 6. SAUVEGARDE CSV
# ==========================================

summary_path = "results/statistics/caption_statistics_summary.csv"

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow(["Metric", "Value"])
    writer.writerow(["Total images", len(data)])
    writer.writerow(["Total captions", len(captions)])
    writer.writerow(["Total word tokens", total_words])
    writer.writerow(["Vocabulary size", len(vocabulary)])
    writer.writerow(["Average words per caption", average_length])
    writer.writerow(["Median words per caption", median_length])
    writer.writerow(["Minimum length", min(lengths)])
    writer.writerow(["Maximum length", max(lengths)])

    writer.writerow([])
    writer.writerow(["Word", "Frequency"])

    for word, count in word_counts.most_common(20):
        writer.writerow([word, count])

    writer.writerow([])
    writer.writerow(["Length Group", "Number", "Percentage"])

    for group, number in length_groups.items():
        writer.writerow([
            group,
            number,
            round(number / len(captions) * 100, 2)
        ])

print("\nCSV saved:", summary_path)

# ==========================================
# 7. GRAPHIQUE
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    list(length_groups.keys()),
    list(length_groups.values())
)

plt.xlabel("Caption Length")
plt.ylabel("Number of Captions")
plt.title("Caption Length Distribution - Knowledge Base")
plt.xticks(rotation=20)
plt.tight_layout()

chart_path = "results/visualizations/caption_length_distribution.png"

plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print("Chart saved:", chart_path)

# ==========================================
# 8. VALIDATION
# ==========================================

assert len(data) == 10000
assert len(captions) == 50000
assert sum(length_groups.values()) == len(captions)

print("\nCaption statistics validated.")
print("CAPTION STATISTICS ANALYSIS COMPLETED!")