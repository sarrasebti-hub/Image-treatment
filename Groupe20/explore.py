import json

path = r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations\captions_train2017.json"

with open(path, "r") as f:
    data = json.load(f)

print("Images:", len(data["images"]))
print("Captions:", len(data["annotations"]))
print("\nExample image entry:")
print(data["images"][0])
print("\nExample caption entry:")
print(data["annotations"][0])