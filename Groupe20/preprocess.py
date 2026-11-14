import json
import re
from collections import defaultdict
from pathlib import Path

ANN_DIR = Path(r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations")
OUT = Path("processed")
OUT.mkdir(exist_ok=True)


def clean(text):
    """Cleaning used ONLY for retrieval/indexing, never for evaluation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)   # drop punctuation
    return re.sub(r"\s+", " ", text).strip()     # normalise whitespace


def build(ann_file, ids_file, keep_clean):
    ids = json.load(open(f"splits/{ids_file}"))
    data = json.load(open(ANN_DIR / ann_file))
    names = {img["id"]: img["file_name"] for img in data["images"]}

    caps = defaultdict(list)
    for a in sorted(data["annotations"], key=lambda a: a["id"]):
        caps[a["image_id"]].append(a["caption"].strip())

    rows = []
    for i in ids:
        raw = caps[i][:5]                         # exactly 5 captions
        assert len(raw) == 5, f"image {i} has fewer than 5 captions"
        row = {"image_id": i, "file_name": names[i], "caption_raw": raw}
        if keep_clean:
            row["caption_clean"] = [clean(c) for c in raw]
        rows.append(row)
    return rows


# knowledge base: raw + clean (clean is used for indexing)
kb = build("captions_train2017.json", "kb_train_ids.json", keep_clean=True)
# validation and test: raw only, these are the evaluation references
val = build("captions_train2017.json", "val_ids.json", keep_clean=False)
test = build("captions_val2017.json", "test_ids.json", keep_clean=False)

test1000 = build("captions_val2017.json", "test_ids_1000.json", keep_clean=False)
json.dump(test1000, open(OUT / "test_1000_references.json", "w"), indent=1)

json.dump(kb, open(OUT / "kb_captions.json", "w"), indent=1)
json.dump(val, open(OUT / "val_references.json", "w"), indent=1)
json.dump(test, open(OUT / "test_references.json", "w"), indent=1)

print("KB:", len(kb), "| val:", len(val), "| test:", len(test), "| test1000:", len(test1000))
print("\nExample:")
print("raw  :", kb[0]["caption_raw"][0])
print("clean:", kb[0]["caption_clean"][0])