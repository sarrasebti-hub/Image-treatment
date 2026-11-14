
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"


def load_data(filename):
    with open(PROCESSED_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_split(name, data):
    images = data["images"]
    annotations = data["annotations"]

    # Extract captions
    captions = [item["caption"] for item in annotations]

    # Number of words in each caption
    words_per_caption = [
        len(re.findall(r"\b\w+\b", caption.lower()))
        for caption in captions
    ]

    # Captions per image
    captions_per_image = Counter(
        item["image_id"] for item in annotations
    )

    print("\n" + "=" * 45)
    print(f"{name.upper()} DATASET")
    print("=" * 45)

    print(f"Number of images: {len(images)}")
    print(f"Number of captions: {len(captions)}")

    print(
        f"Average captions per image: "
        f"{len(captions) / len(images):.2f}"
    )

    print(f"Average words per caption: {np.mean(words_per_caption):.2f}")
    print(f"Minimum words per caption: {min(words_per_caption)}")
    print(f"Maximum words per caption: {max(words_per_caption)}")

    return captions, words_per_caption, images


# Load the three datasets
train_data = load_data("train_captions.json")
val_data = load_data("validation_captions.json")
test_data = load_data("test_captions.json")

# Analyze each split
train_captions, train_lengths, train_images = analyze_split(
    "Train", train_data
)

val_captions, val_lengths, val_images = analyze_split(
    "Validation", val_data
)

test_captions, test_lengths, test_images = analyze_split(
    "Test", test_data
)

# Most frequent words in training captions
all_words = []

for caption in train_captions:
    all_words.extend(
        re.findall(r"\b\w+\b", caption.lower())
    )

word_counts = Counter(all_words)

print("\n" + "=" * 45)
print("TOP 20 MOST FREQUENT WORDS - TRAIN")
print("=" * 45)

for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")


# Save statistics
statistics = {
    "train": {
        "images": len(train_images),
        "captions": len(train_captions),
        "average_words": round(float(np.mean(train_lengths)), 2),
        "min_words": min(train_lengths),
        "max_words": max(train_lengths)
    },
    "validation": {
        "images": len(val_images),
        "captions": len(val_captions),
        "average_words": round(float(np.mean(val_lengths)), 2),
        "min_words": min(val_lengths),
        "max_words": max(val_lengths)
    },
    "test": {
        "images": len(test_images),
        "captions": len(test_captions),
        "average_words": round(float(np.mean(test_lengths)), 2),
        "min_words": min(test_lengths),
        "max_words": max(test_lengths)
    }
}

output_file = DATASET_DIR / "dataset_statistics.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(statistics, f, indent=4, ensure_ascii=False)

print(f"\nStatistics saved to: {output_file}")
