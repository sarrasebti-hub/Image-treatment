import json

path = "annotations/captions_train2017.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]
images = data["images"]

print("Total images in JSON:", len(images))
print("Total captions:", len(annotations))

print("First caption:")
print(annotations[0]["caption"])