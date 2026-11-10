import json

# Load metadata
path = "dataset/dataset/metadata/knowledge_base_metadata.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

bad_cat = 0
bad_bbox = 0
outside = 0
bad_area = 0
total = 0
examples = []

for image in data:
    width = image["width"]
    height = image["height"]

    for obj in image["objects"]:
        total += 1

        x, y, box_width, box_height = obj["bbox"]

        if obj["category"] == "unknown":
            bad_cat += 1

        if (
            box_width <= 0
            or box_height <= 0
            or x < 0
            or y < 0
        ):
            bad_bbox += 1

        if (
            x + box_width > width
            or y + box_height > height
        ):
            outside += 1

        if obj["area"] < 0:
            bad_area += 1

        if (
            box_width <= 0
            or box_height <= 0
            or x < 0
            or y < 0
            or x + box_width > width
            or y + box_height > height
        ) and len(examples) < 5:
            examples.append((image["image_id"], obj))

print("\n========== OBJECT ANNOTATION AUDIT ==========")
print("Total objects:", total)
print("Unknown categories:", bad_cat)
print("Invalid bounding boxes:", bad_bbox)
print("Bounding boxes outside image:", outside)
print("Negative areas:", bad_area)
print("Examples:", examples)
