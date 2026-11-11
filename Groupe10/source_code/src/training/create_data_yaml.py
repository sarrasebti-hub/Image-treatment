import json
from pathlib import Path

ANNOTATIONS_PATH = Path(
    "data/raw/annotations/annotations/instances_train2017.json"
)

OUTPUT_PATH = Path("data/processed/yolo/data.yaml")
DATASET_PATH = OUTPUT_PATH.parent.resolve()

with open(ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

categories = sorted(coco["categories"], key=lambda x: x["id"])
class_names = [category["name"] for category in categories]

yaml_content = f"""path: {DATASET_PATH}
train: images/train
val: images/val

names:
"""

for index, name in enumerate(class_names):
    yaml_content += f"  {index}: '{name}'\n"

OUTPUT_PATH.write_text(yaml_content)

print("YOLO data.yaml created successfully!")
print(f"Classes: {len(class_names)}")
print(f"Dataset path: {DATASET_PATH}")
print(f"Output: {OUTPUT_PATH}")