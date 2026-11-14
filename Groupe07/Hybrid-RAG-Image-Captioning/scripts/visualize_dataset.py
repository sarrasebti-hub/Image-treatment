
import json
import random
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Reproducibility
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATIONS_PATH = (
    BASE_DIR / "dataset" / "processed" / "train_captions.json"
)

IMAGES_DIR = BASE_DIR / "dataset" / "images_train"

OUTPUT_DIR = BASE_DIR / "dataset" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load annotations
with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]


# Group captions by image ID
captions_by_image = {}

for ann in annotations:
    image_id = ann["image_id"]
    captions_by_image.setdefault(image_id, []).append(
        ann["caption"]
    )


# Select 5 reproducible random images
selected_images = random.sample(images, 5)


# Generate visualizations
for i, image_info in enumerate(selected_images, start=1):

    image_id = image_info["id"]
    filename = image_info["file_name"]
    image_path = IMAGES_DIR / filename

    if not image_path.exists():
        print(f"Image not found: {filename}")
        continue

    captions = captions_by_image.get(image_id, [])

    with Image.open(image_path) as image:
        img = image.convert("RGB")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 1.5]}
    )

    # Display image
    axes[0].imshow(img)
    axes[0].set_title(f"Image ID: {image_id}", fontsize=14)
    axes[0].axis("off")

    # Display captions
    axes[1].axis("off")
    axes[1].set_title(
        f"Human Captions ({len(captions)})",
        fontsize=15
    )

    caption_text = "\n\n".join(
        [
            f"{j + 1}. {textwrap.fill(caption, width=55)}"
            for j, caption in enumerate(captions)
        ]
    )

    axes[1].text(
        0.02, 0.95,
        caption_text,
        fontsize=11,
        verticalalignment="top",
        linespacing=1.5
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"image_{i}_{image_id}.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nVisualization completed successfully!")
