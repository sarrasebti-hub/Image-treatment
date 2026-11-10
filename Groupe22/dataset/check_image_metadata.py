import json

# Load metadata
path = "dataset/dataset/metadata/knowledge_base_metadata.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

missing_ids = 0
duplicate_ids = 0
missing_filenames = 0
empty_filenames = 0
invalid_dimensions = 0
duplicate_filenames = 0

image_ids = []
filenames = []

for image in data:
    image_id = image.get("image_id")
    filename = image.get("file_name")
    width = image.get("width")
    height = image.get("height")

    if image_id is None:
        missing_ids += 1
    else:
        image_ids.append(image_id)

    if filename is None:
        missing_filenames += 1
    elif not str(filename).strip():
        empty_filenames += 1
    else:
        filenames.append(filename)

    if (
        not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or width <= 0
        or height <= 0
    ):
        invalid_dimensions += 1

duplicate_ids = len(image_ids) - len(set(image_ids))
duplicate_filenames = len(filenames) - len(set(filenames))

print("\n========== IMAGE METADATA AUDIT ==========")
print("Total images:", len(data))
print("Missing image IDs:", missing_ids)
print("Duplicate image IDs:", duplicate_ids)
print("Missing filenames:", missing_filenames)
print("Empty filenames:", empty_filenames)
print("Duplicate filenames:", duplicate_filenames)
print("Invalid dimensions:", invalid_dimensions)

print("\n========== AUDIT COMPLETED ==========")