import json
import os
from collections import Counter

# Paths
annotations_path = "annotations/captions_train2017.json"
images_path = "train2017"

# Load annotations
with open(annotations_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
captions = data["annotations"]

# Count captions per image
caption_counts = Counter(
    annotation["image_id"]
    for annotation in captions
)

# Get image IDs from JSON
json_image_ids = {img["id"] for img in images}

# Get actual image IDs from folder
folder_image_ids = set()

for filename in os.listdir(images_path):
    if filename.lower().endswith(".jpg"):
        image_id = int(os.path.splitext(filename)[0])
        folder_image_ids.add(image_id)

# Analysis
missing_images = json_image_ids - folder_image_ids
extra_images = folder_image_ids - json_image_ids

images_with_5_captions = sum(
    1 for image_id in json_image_ids
    if caption_counts[image_id] == 5
)

images_with_wrong_captions = sum(
    1 for image_id in json_image_ids
    if caption_counts[image_id] != 5
)

print("========== CAPTION INTEGRITY AUDIT ==========")

print("Images in JSON:", len(json_image_ids))
print("Images in folder:", len(folder_image_ids))
print("Total captions:", len(captions))

print("Images with 5 captions:", images_with_5_captions)
print("Images with incorrect caption count:",
      images_with_wrong_captions)

print("Images missing from folder:", len(missing_images))
print("Images not found in JSON:", len(extra_images))

print("=============================================")