
import json
from pathlib import Path

RAW_IMAGES = Path("data/raw/images")
SPLITS_DIR = Path("data/processed/splits")

# Selected image IDs
kb_ids = set(json.load(open(SPLITS_DIR / "knowledge_base_ids.json")))
val_ids = set(json.load(open(SPLITS_DIR / "validation_ids.json")))
test_ids = set(json.load(open(SPLITS_DIR / "test_ids.json")))

# Load original metadata
annotations_dir = Path("data/raw/annotations/annotations")

train_data = json.load(open(annotations_dir / "captions_train2017.json"))
test_data = json.load(open(annotations_dir / "captions_val2017.json"))

# Selected filenames
train_selected = {
    image["file_name"]
    for image in train_data["images"]
    if image["id"] in kb_ids | val_ids
}

test_selected = {
    image["file_name"]
    for image in test_data["images"]
    if image["id"] in test_ids
}

def preview(folder, selected):
    all_files = {
        path.name for path in folder.iterdir()
        if path.is_file()
    }

    to_delete = all_files - selected

    print(f"\nFolder: {folder}")
    print("Total images:", len(all_files))
    print("Images to keep:", len(all_files & selected))
    print("Images to delete:", len(to_delete))

    return to_delete

train_delete = preview(
    RAW_IMAGES / "train2017",
    train_selected
)

test_delete = preview(
    RAW_IMAGES / "val2017",
    test_selected
)

print("\nDRY RUN ONLY")
print("No files were deleted.")
