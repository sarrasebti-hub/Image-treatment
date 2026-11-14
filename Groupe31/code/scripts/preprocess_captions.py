
import json
import re
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "knowledge_base"
    / "knowledge_base.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "knowledge_base"
    / "knowledge_base_cleaned_captions.jsonl"
)


def clean_caption(caption):
    """Create a normalized copy without changing the original."""
    
    caption = caption.lower()

    # Remove punctuation while keeping letters and numbers
    caption = re.sub(r"[^\w\s]", "", caption)

    # Normalize repeated whitespace
    caption = re.sub(r"\s+", " ", caption).strip()

    return caption


total_images = 0
total_original_captions = 0
total_cleaned_captions = 0

print("Starting caption preprocessing...")

with open(INPUT_FILE, "r", encoding="utf-8") as input_file, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:

    for line in input_file:
        record = json.loads(line)

        original_captions = record["captions"]

        cleaned_captions = [
            clean_caption(caption)
            for caption in original_captions
        ]

        output_record = {
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "original_captions": original_captions,
            "cleaned_captions": cleaned_captions
        }

        output_file.write(
            json.dumps(output_record, ensure_ascii=False) + "\n"
        )

        total_images += 1
        total_original_captions += len(original_captions)
        total_cleaned_captions += len(cleaned_captions)


print("\n========== CAPTION PREPROCESSING REPORT ==========")
print(f"Images processed: {total_images}")
print(f"Original captions preserved: {total_original_captions}")
print(f"Cleaned captions generated: {total_cleaned_captions}")
print(f"Output file: {OUTPUT_FILE}")
print("\nCaption preprocessing completed successfully.")

# Verify the generated dataset
print("\n========== CAPTION DATASET VERIFICATION ==========")

original_file = INPUT_FILE
cleaned_file = OUTPUT_FILE

verified_images = 0
invalid_records = 0
original_count = 0
cleaned_count = 0

with open(original_file, "r", encoding="utf-8") as original, \
     open(cleaned_file, "r", encoding="utf-8") as cleaned:

    for original_line, cleaned_line in zip(original, cleaned):
        original_record = json.loads(original_line)
        cleaned_record = json.loads(cleaned_line)

        if (
            original_record["image_id"] != cleaned_record["image_id"]
            or original_record["file_name"] != cleaned_record["file_name"]
            or original_record["captions"] != cleaned_record["original_captions"]
            or len(cleaned_record["cleaned_captions"]) != 5
            or len(original_record["captions"]) != 5
            or any(not caption for caption in cleaned_record["cleaned_captions"])
        ):
            invalid_records += 1

        else:
            verified_images += 1

        original_count += len(cleaned_record["original_captions"])
        cleaned_count += len(cleaned_record["cleaned_captions"])

print(f"Verified images: {verified_images}")
print(f"Invalid records: {invalid_records}")
print(f"Original captions: {original_count}")
print(f"Cleaned captions: {cleaned_count}")

if verified_images == 10000 and invalid_records == 0:
    print("\nVERIFICATION PASSED")
else:
    print("\nVERIFICATION FAILED")
