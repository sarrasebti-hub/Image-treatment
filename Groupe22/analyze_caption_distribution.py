import json
from collections import Counter

path = "annotations/captions_train2017.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

caption_counts = Counter(
    annotation["image_id"]
    for annotation in data["annotations"]
)

distribution = Counter(caption_counts.values())

print("===== CAPTION DISTRIBUTION =====")

for number, count in sorted(distribution.items()):
    print(f"{number} captions: {count} images")

print("================================")

wrong_images = [
    (image_id, count)
    for image_id, count in caption_counts.items()
    if count != 5
]

print("\nImages with incorrect caption count:")
for image_id, count in wrong_images[:10]:
    print(f"Image ID: {image_id} | Captions: {count}")

print("\nTotal incorrect images:", len(wrong_images))