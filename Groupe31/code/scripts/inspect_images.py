
import json
from collections import Counter
from pathlib import Path

from PIL import Image

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = PROJECT_ROOT / "results" / "knowledge_base" / "knowledge_base.jsonl"
IMAGE_DIR = PROJECT_ROOT / "datasets" / "train2017"

# Statistics
format_counts = Counter()
mode_counts = Counter()
dimension_counts = Counter()
exif_orientation_counts = Counter()

total_images = 0
unreadable_images = 0
images_with_exif_orientation = 0

print("Inspecting knowledge-base images...")

with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        image_path = IMAGE_DIR / record["file_name"]

        try:
            with Image.open(image_path) as img:
                total_images += 1

                format_counts[img.format] += 1
                mode_counts[img.mode] += 1
                dimension_counts[img.size] += 1

                orientation = img.getexif().get(274)

                if orientation is not None:
                    images_with_exif_orientation += 1
                    exif_orientation_counts[orientation] += 1

        except Exception as error:
            unreadable_images += 1
            print(f"Could not read {record['file_name']}: {error}")

print("\n========== IMAGE INSPECTION REPORT ==========")

print(f"Images inspected: {total_images}")
print(f"Unreadable images: {unreadable_images}")

print("\nImage formats:")
for image_format, count in format_counts.most_common():
    print(f"{image_format}: {count}")

print("\nImage colour modes:")
for mode, count in mode_counts.most_common():
    print(f"{mode}: {count}")

print("\nMost common dimensions:")
for dimensions, count in dimension_counts.most_common(10):
    print(f"{dimensions[0]} x {dimensions[1]}: {count}")

print("\nEXIF orientation:")
print(f"Images with orientation metadata: {images_with_exif_orientation}")

for orientation, count in exif_orientation_counts.items():
    print(f"Orientation {orientation}: {count}")

print("\nInspection completed. Original images were not modified.")
