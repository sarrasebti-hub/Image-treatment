
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = PROJECT_ROOT / "results" / "knowledge_base" / "knowledge_base.jsonl"
TABLE_DIR = PROJECT_ROOT / "results" / "statistics" / "tables"
GRAPH_DIR = PROJECT_ROOT / "results" / "statistics" / "graphs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Store caption statistics
caption_lengths = []
all_words = []

# Read captions from the Knowledge Base
with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        for caption in record["captions"]:

            # Count whitespace-separated tokens
            whitespace_tokens = len(caption.split())

            # Extract words, ignoring punctuation
            words = re.findall(r"\b[\w']+\b", caption.lower())

            caption_lengths.append({
                "Image ID": record["image_id"],
                "Caption": caption,
                "Word Count": len(words),
                "Token Count": whitespace_tokens
            })

            all_words.extend(words)

# Create DataFrame
df = pd.DataFrame(caption_lengths)

# Calculate required statistics
total_captions = len(df)
average_words = df["Word Count"].mean()
average_tokens = df["Token Count"].mean()

# Vocabulary: distinct lowercase words
vocabulary_size = len(set(all_words))

# Summary table
summary = pd.DataFrame({
    "Metric": [
        "Total Captions",
        "Average Caption Length (Words)",
        "Average Caption Length (Tokens)",
        "Vocabulary Size"
    ],
    "Value": [
        total_captions,
        round(average_words, 2),
        round(average_tokens, 2),
        vocabulary_size
    ]
})

summary.to_csv(
    TABLE_DIR / "caption_statistics.csv",
    index=False
)

# Save individual caption lengths
df.to_csv(
    TABLE_DIR / "caption_lengths.csv",
    index=False
)

# Caption length histogram
plt.figure(figsize=(10, 6))

plt.hist(
    df["Word Count"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Number of Words per Caption")
plt.ylabel("Number of Captions")
plt.title("Distribution of Caption Lengths")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "caption_length_histogram.png",
    dpi=300
)

plt.close()

# Display results
print("========== CAPTION STATISTICS ==========")
print(f"Total captions: {total_captions}")
print(f"Average caption length (words): {average_words:.2f}")
print(f"Average caption length (tokens): {average_tokens:.2f}")
print(f"Vocabulary size: {vocabulary_size}")

print("\nSummary table:")
print(summary.to_string(index=False))

print("\nCSV tables and histogram saved successfully.")
