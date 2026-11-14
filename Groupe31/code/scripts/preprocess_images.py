
import json
from pathlib import Path
from PIL import Image

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KB_FILE = (
    PROJECT_ROOT
    / "results"
    / "knowledge_base"
    / "knowledge_base.jsonl"
)

SOURCE_DIR = PROJECT_ROOT / "datasets" / "train2017"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "preprocessed_images"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_images = 0
converted_images = 0
failed_images = 0

print("Starting image preprocessing...")

with open(KB_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        filename = record["file_name"]

        source_path = SOURCE_DIR / filename
        output_path = OUTPUT_DIR / filename

        try:
            with Image.open(source_path) as img:

                original_mode = img.mode

                # Convert image to RGB
                rgb_image = img.convert("RGB")

                # Preserve original dimensions
                if rgb_image.size != img.size:
                    raise ValueError("Image dimensions changed unexpectedly")

                # Save processed image
                rgb_image.save(
                    output_path,
                    format="JPEG",
                    quality=95
                )

                total_images += 1

                if original_mode != "RGB":
                    converted_images += 1

        except Exception as error:
            failed_images += 1
            print(f"Error processing {filename}: {error}")


print("\n========== IMAGE PREPROCESSING REPORT ==========")
print(f"Images processed: {total_images}")
print(f"Non-RGB images converted: {converted_images}")
print(f"Failed images: {failed_images}")
print(f"Output directory: {OUTPUT_DIR}")

print("\nImage preprocessing completed.")
