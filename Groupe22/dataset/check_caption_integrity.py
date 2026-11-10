import json
from collections import Counter

# Load metadata
path = "dataset/dataset/metadata/knowledge_base_metadata.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

total_images = len(data)
total_captions = 0
images_without_captions = []
images_with_wrong_count = []
missing_caption_fields = 0
empty_captions = 0
caption_duplicates = 0
caption_image_errors = 0

caption_ids = set()

for image in data:
    image_id = image["image_id"]
    captions = image.get("captions", [])

    total_captions += len(captions)

    if len(captions) == 0:
        images_without_captions.append(image_id)

    if len(captions) != 5:
        images_with_wrong_count.append(
            (image_id, len(captions))
        )

    for caption in captions:
        if not isinstance(caption, str):
            missing_caption_fields += 1
        elif not caption.strip():
            empty_captions += 1

    duplicates = len(captions) - len(set(captions))
    caption_duplicates += duplicates

# Load source cleaned file
source_path = (
    "dataset/dataset/splits/knowledge_base/"
    "knowledge_base_cleaned.json"
)

with open(source_path, "r", encoding="utf-8") as f:
    source = json.load(f)

source_captions = Counter(
    (a["image_id"], a["caption"])
    for a in source["annotations"]
)

metadata_captions = Counter(
    (image["image_id"], caption)
    for image in data
    for caption in image["captions"]
)

missing_from_metadata = source_captions - metadata_captions
extra_in_metadata = metadata_captions - source_captions

print("\n========== CAPTION INTEGRITY AUDIT ==========")
print("Total images:", total_images)
print("Total captions:", total_captions)
print("Images without captions:", len(images_without_captions))
print("Images with count different from 5:", len(images_with_wrong_count))
print("Non-string captions:", missing_caption_fields)
print("Empty captions:", empty_captions)
print("Repeated captions within images:", caption_duplicates)
print("Source captions missing from metadata:", sum(missing_from_metadata.values()))
print("Extra captions in metadata:", sum(extra_in_metadata.values()))

print("\nExamples of images with incorrect caption count:")
print(images_with_wrong_count[:10])

print("\n========== AUDIT COMPLETED ==========")