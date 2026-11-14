
import json
import re
from collections import Counter
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

# Lexical indicators
categories = {
    "Colour": {
        "red": ["red"],
        "blue": ["blue"],
        "green": ["green"],
        "yellow": ["yellow"],
        "black": ["black"],
        "white": ["white"],
        "brown": ["brown"],
        "gray": ["gray", "grey"],
        "orange": ["orange"],
        "pink": ["pink"],
        "purple": ["purple"],
    },

    "Shape": {
        "round": ["round", "rounded"],
        "square": ["square"],
        "rectangular": ["rectangular"],
        "oval": ["oval"],
        "triangular": ["triangular"],
        "circular": ["circular"],
    },

    "Context": {
        "indoor": [
            "indoors", "inside", "room", "kitchen",
            "bedroom", "living room", "office"
        ],
        "outdoor": [
            "outdoors", "outside", "street", "road",
            "park", "beach", "field"
        ],
    },

    "Time of Day": {
        "daytime": ["daytime", "during the day", "daylight"],
        "nighttime": ["nighttime", "at night", "during the night"],
    }
}


def contains_term(text, term):
    """Check for a complete word or phrase."""
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text) is not None


# Count caption-level mentions
counts = Counter()
captions_analyzed = 0

print("Analyzing caption attributes and context mentions...")

with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        for caption in record["captions"]:
            captions_analyzed += 1
            text = caption.lower()

            for category, values in categories.items():
                for label, terms in values.items():

                    if any(contains_term(text, term) for term in terms):
                        counts[(category, label)] += 1


# Build results table
rows = []

for category, values in categories.items():
    for label in values:
        frequency = counts[(category, label)]

        rows.append({
            "Category": category,
            "Label": label,
            "Caption Count": frequency,
            "Percentage": round(
                frequency / captions_analyzed * 100, 2
            )
        })

df = pd.DataFrame(rows)

# Save complete table
df.to_csv(
    TABLE_DIR / "attributes_context_mentions.csv",
    index=False
)

# Generate one chart for each category
for category in categories:

    category_df = df[df["Category"] == category]

    plt.figure(figsize=(10, 6))

    plt.bar(
        category_df["Label"],
        category_df["Caption Count"]
    )

    plt.xlabel("Mention")
    plt.ylabel("Number of Captions")
    plt.title(f"{category} Mentions in COCO Captions")
    plt.xticks(rotation=35)
    plt.tight_layout()

    filename = category.lower().replace(" ", "_") + "_mentions.png"

    plt.savefig(
        GRAPH_DIR / filename,
        dpi=300
    )

    plt.close()


# Display results
print("\n========== ATTRIBUTES AND CONTEXT ANALYSIS ==========")
print(f"Captions analyzed: {captions_analyzed}")
print("\nCaption-level mention counts:")
print(df.to_string(index=False))

print("\nCSV table and four graphs saved successfully.")
print("Analysis completed.")
