
import json
import os
import hashlib
from collections import Counter
from PIL import Image

IMAGE_DIR = "data/raw/images/train2017"
ANNOTATIONS_FILE = "data/raw/annotations/annotations/captions_train2017.json"

# Load COCO annotations
with open(ANNOTATIONS_FILE, "r") as file:
    data = json.load(file)

images = data["images"]
annotations = data["annotations"]

image_files = os.listdir(IMAGE_DIR)

# Check image IDs
image_ids = {image["id"] for image in images}
caption_image_ids = {annotation["image_id"] for annotation in annotations}

invalid_captions = caption_image_ids - image_ids
print("Captions linked to non-existing images:", len(invalid_captions))

# Check filenames
json_filenames = {image["file_name"] for image in images}
folder_filenames = set(image_files)

missing_images = json_filenames - folder_filenames
extra_images = folder_filenames - json_filenames

print("Images missing from folder:", len(missing_images))
print("Extra images in folder:", len(extra_images))

print("Number of image files:", len(image_files))
print("Number of images:", len(images))
print("Number of captions:", len(annotations))

# Check corrupted and duplicate images
corrupted_images = []
image_hashes = {}
duplicate_images = []

print("\nStarting image integrity and duplicate checks...")

for index, filename in enumerate(image_files, start=1):

    image_path = os.path.join(IMAGE_DIR, filename)

    try:
        with Image.open(image_path) as img:
            img.load()

        with open(image_path, "rb") as file:
            image_hash = hashlib.md5(file.read()).hexdigest()

        if image_hash in image_hashes:
            duplicate_images.append(filename)
        else:
            image_hashes[image_hash] = filename

    except Exception as error:
        corrupted_images.append(filename)

    # Display progress every 5000 images
    if index % 5000 == 0:
        print(f"Checked {index}/{len(image_files)} images...", flush=True)

print("\nNumber of corrupted images:", len(corrupted_images))
print("Number of duplicate images:", len(duplicate_images))

# Check captions per image
captions_per_image = Counter(
    annotation["image_id"] for annotation in annotations
)

images_with_wrong_caption_count = {
    image_id: count
    for image_id, count in captions_per_image.items()
    if count != 5
}

print(
    "Images with exactly 5 captions:",
    len(images) - len(images_with_wrong_caption_count)
)

print(
    "Images with a different number of captions:",
    len(images_with_wrong_caption_count)
)

if images_with_wrong_caption_count:
    print(
        "Examples:",
        list(images_with_wrong_caption_count.items())[:10]
    )

# Count how many images have each caption count
caption_count_distribution = Counter(
    captions_per_image.values()
)

print("\nCaption count distribution:")

for count, number_of_images in sorted(caption_count_distribution.items()):
    print(f"{number_of_images} images have {count} captions")

# Find images with fewer than 5 captions
images_with_fewer_captions = {
    image_id: count
    for image_id, count in images_with_wrong_caption_count.items()
    if count < 5
}

# Find images with more than 5 captions
images_with_extra_captions = {
    image_id: count
    for image_id, count in images_with_wrong_caption_count.items()
    if count > 5
}

print("\nImages with fewer than 5 captions:",
      len(images_with_fewer_captions))

print("Images with more than 5 captions:",
      len(images_with_extra_captions))