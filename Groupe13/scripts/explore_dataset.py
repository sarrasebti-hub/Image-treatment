
from pathlib import Path
import json
import re
from collections import Counter
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "exploration"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPTIONS_FILE = ANNOTATIONS_DIR / "captions_train2017.json"
INSTANCES_FILE = ANNOTATIONS_DIR / "instances_train2017.json"


# ==========================================================
# 2. CHARGEMENT DES FICHIERS JSON
# ==========================================================

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    print(f"Loading: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================================
# 3. EXPLORATION DES CAPTIONS
# ==========================================================

def explore_captions(data):

    print("\n========== CAPTION EXPLORATION ==========")

    captions = [
        annotation["caption"]
        for annotation in data["annotations"]
    ]

    if not captions:
        raise ValueError("No captions found.")

    # Longueur de chaque caption en nombre de mots
    caption_lengths = [
        len(re.findall(r"\b\w+\b", caption))
        for caption in captions
    ]

    # Extraction des mots
    all_words = []

    for caption in captions:

        words = re.findall(
            r"[a-zA-Z]+",
            caption.lower()
        )

        all_words.extend(words)

    # Fréquence des mots
    word_counter = Counter(all_words)

    # Taille du vocabulaire
    vocabulary_size = len(word_counter)

    # Top 30 mots
    top_words = word_counter.most_common(30)

    # Statistiques
    statistics = {
        "total_captions": len(captions),
        "total_words": len(all_words),
        "vocabulary_size": vocabulary_size,
        "average_caption_length": round(
            sum(caption_lengths) / len(caption_lengths), 2
        ),
        "minimum_caption_length": min(caption_lengths),
        "maximum_caption_length": max(caption_lengths),
        "median_caption_length": float(
            pd.Series(caption_lengths).median()
        )
    }

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\nTop 30 frequent words:")

    for word, frequency in top_words:
        print(f"{word}: {frequency}")

    # Sauvegarder les fréquences des mots
    words_df = pd.DataFrame(
        top_words,
        columns=["word", "frequency"]
    )

    words_df.to_csv(
        OUTPUT_DIR / "top_words.csv",
        index=False
    )

    # Graphique 1 : distribution des longueurs
    plt.figure(figsize=(10, 5))

    plt.hist(
        caption_lengths,
        bins=30,
        edgecolor="black"
    )

    plt.title("Distribution of Caption Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Captions")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "caption_lengths.png",
        dpi=300
    )

    plt.close()

    # Graphique 2 : mots les plus fréquents
    top_15 = words_df.head(15)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_15["word"],
        top_15["frequency"]
    )

    plt.title("Top 15 Most Frequent Words")
    plt.xlabel("Frequency")
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_words.png",
        dpi=300
    )

    plt.close()

    return statistics


# ==========================================================
# 4. EXPLORATION DES OBJETS
# ==========================================================

def explore_objects(data):

    print("\n========== OBJECT EXPLORATION ==========")

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    annotations = data["annotations"]
    images = data["images"]

    # Fréquence des catégories
    category_counter = Counter()

    # Nombre d'objets par image
    objects_per_image = Counter()

    # Catégories présentes dans chaque image
    image_categories = {
        image["id"]: set()
        for image in images
    }

    for annotation in annotations:

        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        category_name = categories.get(
            category_id,
            "unknown"
        )

        category_counter[category_name] += 1

        objects_per_image[image_id] += 1

        if image_id in image_categories:
            image_categories[image_id].add(category_name)

    total_images = len(images)
    total_annotations = len(annotations)

    average_objects = (
        total_annotations / total_images
        if total_images > 0
        else 0
    )

    # Top 50 catégories
    top_50 = category_counter.most_common(50)

    statistics = {
        "total_images": total_images,
        "total_object_annotations": total_annotations,
        "number_of_categories": len(category_counter),
        "average_objects_per_image": round(
            average_objects, 2
        )
    }

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\nTop 50 object categories:")

    for category, frequency in top_50:
        print(f"{category}: {frequency}")

    # Sauvegarder les catégories
    categories_df = pd.DataFrame(
        top_50,
        columns=["category", "frequency"]
    )

    categories_df.to_csv(
        OUTPUT_DIR / "top_50_objects.csv",
        index=False
    )

    # Graphique : Top 20 catégories
    top_20 = categories_df.head(20)

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_20["category"],
        top_20["frequency"]
    )

    plt.title("Top 20 Object Categories")
    plt.xlabel("Number of Annotations")
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_20_objects.png",
        dpi=300
    )

    plt.close()

    # Graphique : objets par image
    object_counts = [
        objects_per_image.get(image["id"], 0)
        for image in images
    ]

    plt.figure(figsize=(10, 5))

    plt.hist(
        object_counts,
        bins=30,
        edgecolor="black"
    )

    plt.title("Distribution of Objects per Image")
    plt.xlabel("Number of Objects")
    plt.ylabel("Number of Images")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "objects_per_image.png",
        dpi=300
    )

    plt.close()

    return statistics, image_categories


# ==========================================================
# 5. CO-OCCURRENCE DES OBJETS
# ==========================================================

def explore_cooccurrence(image_categories):

    print("\n========== OBJECT CO-OCCURRENCE ==========")

    pair_counter = Counter()

    for categories in image_categories.values():

        unique_categories = sorted(categories)

        for pair in combinations(unique_categories, 2):
            pair_counter[pair] += 1

    top_pairs = pair_counter.most_common(20)

    for pair, frequency in top_pairs:
        print(
            f"{pair[0]} + {pair[1]}: {frequency}"
        )

    pairs_df = pd.DataFrame(
        [
            {
                "object_1": pair[0],
                "object_2": pair[1],
                "frequency": frequency
            }
            for pair, frequency in top_pairs
        ],
        columns=["object_1", "object_2", "frequency"]
    )

    pairs_df.to_csv(
        OUTPUT_DIR / "object_cooccurrence.csv",
        index=False
    )


# ==========================================================
# 6. SAUVEGARDE DU RAPPORT GLOBAL
# ==========================================================

def save_summary(caption_stats, object_stats):

    report = {
        "captions": caption_stats,
        "objects": object_stats
    }

    output_file = OUTPUT_DIR / "exploration_summary.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nSummary saved: {output_file}")


# ==========================================================
# 7. PROGRAMME PRINCIPAL
# ==========================================================

def main():

    print("=" * 55)
    print("COCO DATASET EXPLORATION")
    print("=" * 55)

    # Charger les fichiers
    captions_data = load_json(CAPTIONS_FILE)
    instances_data = load_json(INSTANCES_FILE)

    # Explorer les captions
    caption_stats = explore_captions(captions_data)

    # Explorer les objets
    object_stats, image_categories = explore_objects(
        instances_data
    )

    # Étudier les co-occurrences
    explore_cooccurrence(image_categories)

    # Sauvegarder le rapport
    save_summary(
        caption_stats,
        object_stats
    )

    print("\n" + "=" * 55)
    print("DATA EXPLORATION COMPLETED SUCCESSFULLY")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
