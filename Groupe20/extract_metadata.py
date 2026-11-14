import json
from collections import Counter, defaultdict
from pathlib import Path

ANN_DIR = Path(r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations")
OUT = Path("metadata")
OUT.mkdir(exist_ok=True)


def load_instances(name):
    data = json.load(open(ANN_DIR / name))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    per_image = defaultdict(list)
    for a in data["annotations"]:
        per_image[a["image_id"]].append({
            "category": cats[a["category_id"]],
            "bbox": [round(x, 1) for x in a["bbox"]],   # x, y, width, height
            "area": round(a["area"], 1),
            "iscrowd": a["iscrowd"],
        })
    return per_image


def build(per_image, ids_file, source):
    ids = json.load(open(f"splits/{ids_file}"))
    rows = {}
    for i in ids:
        objs = per_image.get(i, [])
        rows[str(i)] = {
            "objects": objs,
            "category_counts": dict(Counter(o["category"] for o in objs)),
            "n_objects": len(objs),
        }
    return {"source": source, "n_images": len(ids), "images": rows}


train_inst = load_instances("instances_train2017.json")
val_inst = load_instances("instances_val2017.json")

kb = build(train_inst, "kb_train_ids.json", "COCO instances_train2017 annotations")
val = build(train_inst, "val_ids.json", "COCO instances_train2017 annotations")
test = build(val_inst, "test_ids.json", "COCO instances_val2017 annotations")

json.dump(kb, open(OUT / "kb_metadata.json", "w"))
json.dump(val, open(OUT / "val_objects_gt.json", "w"))
json.dump(test, open(OUT / "test_objects_gt.json", "w"))

no_obj = sum(v["n_objects"] == 0 for v in kb["images"].values())
print("KB images:", kb["n_images"], "| with no objects:", no_obj)
print("Example:", json.dumps(next(iter(kb["images"].items())))[:300])