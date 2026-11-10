import os
import json
import re

# =====================================
# COCO 2017 - CAPTION PREPROCESSING
# Groupe 22 - Corrective RAG
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(BASE_DIR, "splits")

SPLITS = ["knowledge_base", "validation", "test"]


# =====================================
# CLEAN CAPTION
# =====================================

def clean_caption(text):
    """
    Clean and normalize a caption.
    """

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# =====================================
# PROCESS ONE SPLIT
# =====================================

def process_split(split_name):

    input_path = os.path.join(
        SPLITS_DIR,
        split_name,
        f"{split_name}.json"
    )

    output_path = os.path.join(
        SPLITS_DIR,
        split_name,
        f"{split_name}_cleaned.json"
    )

    print(f"\nProcessing: {split_name}")

    # Load JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data["annotations"])

    # Clean captions
    for annotation in data["annotations"]:

        if "caption" in annotation:
            annotation["caption"] = clean_caption(
                annotation["caption"]
            )

    # Remove empty captions
    data["annotations"] = [
        annotation
        for annotation in data["annotations"]
        if annotation.get("caption", "").strip()
    ]

    cleaned_count = len(data["annotations"])

    # Save cleaned file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Original captions: {original_count}")
    print(f"Cleaned captions: {cleaned_count}")
    print(f"Saved to: {output_path}")


# =====================================
# MAIN
# =====================================

print("=====================================")
print("CAPTION PREPROCESSING")
print("=====================================")

for split in SPLITS:
    process_split(split)

print("\n=====================================")
print("Caption preprocessing completed!")
print("=====================================")