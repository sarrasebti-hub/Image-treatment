
import json

path = "annotations/captions_val2017.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("===== VALIDATION CAPTIONS AUDIT =====")
print("Total images:", len(data["images"]))
print("Total captions:", len(data["annotations"]))
print("First caption:", data["annotations"][0]["caption"])
print("=====================================")

