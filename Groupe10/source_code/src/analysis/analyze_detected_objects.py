import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path("data/processed")
SPLITS = ["knowledge_base", "validation", "test"]


def analyze_split(split):
    metadata_path = DATA_DIR / split / "metadata.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    object_counter = Counter()
    images_per_object = defaultdict(set)

    total_objects = 0
    images_with_objects = 0
    processed_images = 0

    for item in metadata:
        if "objects" not in item:
            continue

        processed_images += 1
        image_id = item["image_id"]
        objects = item["objects"]

        if objects:
            images_with_objects += 1

        total_objects += len(objects)

        for obj in objects:
            name = obj["name"]
            object_counter[name] += 1
            images_per_object[name].add(image_id)

    results = {
        "split": split,
        "total_images_in_split": len(metadata),
        "processed_images": processed_images,
        "images_with_objects": images_with_objects,
        "total_objects": total_objects,
        "average_objects_per_processed_image": round(
            total_objects / processed_images, 2
        ) if processed_images else 0,
        "object_frequency": dict(object_counter.most_common()),
        "images_per_object": {
            name: len(image_ids)
            for name, image_ids in images_per_object.items()
        }
    }

    print(f"\n===== {split.upper()} =====")
    print(f"Total images in split: {len(metadata)}")
    print(f"Images processed: {processed_images}")
    print(f"Images with objects: {images_with_objects}")
    print(f"Total detected objects: {total_objects}")
    print(
        f"Average objects per processed image: "
        f"{results['average_objects_per_processed_image']}"
    )

    print("\nTop 10 detected objects:")
    for name, count in object_counter.most_common(10):
        print(
            f"{name}: {count} detections "
            f"in {len(images_per_object[name])} images"
        )

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{split}_object_analysis.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    for split in SPLITS:
        analyze_split(split)
