
import json
from pathlib import Path
from PIL import Image
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

SPLITS_PATH = BASE_DIR / "dataset" / "dataset_splits.json"
IMAGES_DIR = BASE_DIR / "dataset" / "images_train"
OUTPUT_DIR = BASE_DIR / "dataset" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SPLITS_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)

train_ids = set(splits["train"])
validation_ids = set(splits["validation"])

all_ids = train_ids | validation_ids

dimensions = Counter()
widths = []
heights = []
ratios = []

for image_id in tqdm(all_ids, desc="Analyzing image dimensions"):
    filename = f"{image_id:012d}.jpg"
    image_path = IMAGES_DIR / filename

    if not image_path.exists():
        continue

    try:
        with Image.open(image_path) as img:
            width, height = img.size

        dimensions[f"{width}x{height}"] += 1
        widths.append(width)
        heights.append(height)
        ratios.append(width / height)

    except Exception as error:
        print(f"Error with {filename}: {error}")

print("\nIMAGE DIMENSIONS ANALYSIS")
print("-------------------------")
print(f"Total analyzed images: {len(widths)}")
print(f"Average width: {sum(widths) / len(widths):.2f}")
print(f"Average height: {sum(heights) / len(heights):.2f}")
print(f"Minimum width: {min(widths)}")
print(f"Maximum width: {max(widths)}")
print(f"Minimum height: {min(heights)}")
print(f"Maximum height: {max(heights)}")
print(f"Average aspect ratio: {sum(ratios) / len(ratios):.2f}")

print("\nTOP 10 IMAGE DIMENSIONS")
for dimension, count in dimensions.most_common(10):
    print(f"{dimension}: {count}")

# Width distribution
plt.figure(figsize=(10, 5))
plt.hist(widths, bins=30)
plt.title("Distribution of Image Widths")
plt.xlabel("Width (pixels)")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "image_width_distribution.png", dpi=150)
plt.close()

# Height distribution
plt.figure(figsize=(10, 5))
plt.hist(heights, bins=30)
plt.title("Distribution of Image Heights")
plt.xlabel("Height (pixels)")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "image_height_distribution.png", dpi=150)
plt.close()

print("\nCharts saved successfully!")
