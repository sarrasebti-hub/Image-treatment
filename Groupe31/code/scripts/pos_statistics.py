
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import spacy

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = PROJECT_ROOT / "results" / "knowledge_base" / "knowledge_base.jsonl"
TABLE_DIR = PROJECT_ROOT / "results" / "statistics" / "tables"
GRAPH_DIR = PROJECT_ROOT / "results" / "statistics" / "graphs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load English NLP model
print("Loading spaCy English model...")
nlp = spacy.load("en_core_web_sm")

# Read all captions
captions = []

with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        captions.extend(record["captions"])

print(f"Total captions loaded: {len(captions)}")

# Count grammatical categories
pos_counts = Counter()

print("Analyzing captions...")

for doc in nlp.pipe(captions, batch_size=500):
    for token in doc:
        if not token.is_space and not token.is_punct:
            pos_counts[token.pos_] += 1

# Create results table
total_tokens = sum(pos_counts.values())

df = pd.DataFrame(
    pos_counts.most_common(),
    columns=["Part of Speech", "Frequency"]
)

df["Percentage"] = (
    df["Frequency"] / total_tokens * 100
).round(2)

# Save CSV table
df.to_csv(
    TABLE_DIR / "pos_distribution.csv",
    index=False
)

# Generate POS distribution chart
plt.figure(figsize=(11, 6))

plt.bar(
    df["Part of Speech"],
    df["Frequency"]
)

plt.xlabel("Part of Speech")
plt.ylabel("Number of Tokens")
plt.title("Part-of-Speech Distribution in COCO Captions")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "pos_distribution.png",
    dpi=300
)

plt.close()

# Display results
print("\n========== POS DISTRIBUTION ==========")
print(f"Total captions analyzed: {len(captions)}")
print(f"Total linguistic tokens: {total_tokens}")
print(f"Unique POS categories: {len(pos_counts)}")
print()
print(df.to_string(index=False))

print("\nCSV table and graph saved successfully.")
