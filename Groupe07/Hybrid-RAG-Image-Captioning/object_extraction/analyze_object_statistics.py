
import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "object_extraction" / "results"
OUTPUT_DIR = BASE_DIR / "object_extraction" / "statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_split(split_name):

    input_file = RESULTS_DIR / f"{split_name}_detections.jsonl"

    print(f"\nAnalyzing {split_name}...")

    object_counts = Counter()
    confidence_by_class = defaultdict(list)
    objects_per_image = []
    all_confidences = []

    total_images = 0
    empty_images = 0
    total_objects = 0

    with open(input_file, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            total_images += 1

            detected_objects = record["objects"]
            number_of_objects = len(detected_objects)

            objects_per_image.append(number_of_objects)

            if number_of_objects == 0:
                empty_images += 1

            total_objects += number_of_objects

            for obj in detected_objects:

                label = obj["label"]
                confidence = obj["confidence"]

                object_counts[label] += 1
                confidence_by_class[label].append(confidence)
                all_confidences.append(confidence)

    # Calculate average confidence for each category
    class_statistics = {}

    for label, scores in confidence_by_class.items():

        class_statistics[label] = {
            "count": len(scores),
            "average_confidence": round(
                sum(scores) / len(scores), 4
            )
        }

    average_objects = (
        total_objects / total_images if total_images else 0
    )

    average_confidence = (
        sum(all_confidences) / len(all_confidences)
        if all_confidences else 0
    )

    # Build report
    statistics = {
        "split": split_name,
        "total_images": total_images,
        "total_objects": total_objects,
        "unique_categories": len(object_counts),
        "average_objects_per_image": round(average_objects, 4),
        "empty_images": empty_images,
        "images_with_objects": total_images - empty_images,
        "average_confidence": round(average_confidence, 4),
        "top_20_categories": dict(object_counts.most_common(20)),
        "class_statistics": class_statistics
    }

    # Save JSON statistics
    json_path = OUTPUT_DIR / f"{split_name}_statistics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=4, ensure_ascii=False)

    # Chart 1: Top 20 object categories
    top_categories = object_counts.most_common(20)

    if top_categories:

        labels = [item[0] for item in top_categories]
        counts = [item[1] for item in top_categories]

        plt.figure(figsize=(12, 7))
        plt.barh(labels[::-1], counts[::-1])
        plt.title(f"Top 20 Detected Objects - {split_name.title()}")
        plt.xlabel("Number of detections")
        plt.ylabel("Object category")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{split_name}_top_objects.png",
            dpi=150
        )
        plt.close()

    # Chart 2: Objects per image
    plt.figure(figsize=(10, 6))
    plt.hist(
        objects_per_image,
        bins=20,
        edgecolor="black"
    )
    plt.title(f"Objects per Image - {split_name.title()}")
    plt.xlabel("Number of detected objects")
    plt.ylabel("Number of images")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{split_name}_objects_distribution.png",
        dpi=150
    )
    plt.close()

    # Chart 3: Confidence distribution
    if all_confidences:

        plt.figure(figsize=(10, 6))
        plt.hist(
            all_confidences,
            bins=20,
            edgecolor="black"
        )
        plt.title(f"Confidence Score Distribution - {split_name.title()}")
        plt.xlabel("Confidence score")
        plt.ylabel("Number of detections")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{split_name}_confidence_distribution.png",
            dpi=150
        )
        plt.close()

    # Display results
    print("\n========== STATISTICS ==========")
    print(f"Split: {split_name}")
    print(f"Total images: {total_images}")
    print(f"Total detected objects: {total_objects}")
    print(f"Unique object categories: {len(object_counts)}")
    print(f"Average objects per image: {average_objects:.2f}")
    print(f"Images without detections: {empty_images}")
    print(f"Images with detections: {total_images - empty_images}")
    print(f"Average confidence: {average_confidence:.4f}")

    print("\nTOP 20 OBJECT CATEGORIES")

    for label, count in object_counts.most_common(20):
        print(f"{label}: {count}")

    print(f"\nStatistics saved: {json_path}")


# Analyze both datasets
analyze_split("train")
analyze_split("validation")

print("\n====================================")
print("OBJECT STATISTICS ANALYSIS COMPLETED")
print("====================================")
print(f"All results saved in: {OUTPUT_DIR}")
