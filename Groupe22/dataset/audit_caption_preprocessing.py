import json
import re
import os

BASE = "dataset/dataset/splits"

SPLITS = ["knowledge_base", "validation", "test"]


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


for split in SPLITS:

    original_path = os.path.join(
        BASE, split, f"{split}.json"
    )

    cleaned_path = os.path.join(
        BASE, split, f"{split}_cleaned.json"
    )

    with open(original_path, encoding="utf-8") as f:
        original = json.load(f)

    with open(cleaned_path, encoding="utf-8") as f:
        cleaned = json.load(f)

    original_annotations = original["annotations"]
    cleaned_annotations = cleaned["annotations"]

    original_images = original["images"]
    cleaned_images = cleaned["images"]

    expected = [
        normalize(x["caption"])
        for x in original_annotations
        if x.get("caption", "").strip()
    ]

    actual = [
        x.get("caption", "")
        for x in cleaned_annotations
    ]

    print("\n==========", split.upper(), "==========")

    print("Original images:", len(original_images))
    print("Cleaned images:", len(cleaned_images))

    print("Original captions:", len(original_annotations))
    print("Cleaned captions:", len(cleaned_annotations))

    print(
        "Image IDs unchanged:",
        [x["id"] for x in original_images]
        == [x["id"] for x in cleaned_images]
    )

    print(
        "Caption normalization correct:",
        expected == actual
    )

    print(
        "Annotations removed:",
        len(original_annotations) - len(cleaned_annotations)
    )

    print(
        "Caption IDs/order unchanged:",
        [
            (x.get("id"), x.get("image_id"))
            for x in original_annotations
            if x.get("caption", "").strip()
        ]
        ==
        [
            (x.get("id"), x.get("image_id"))
            for x in cleaned_annotations
        ]
    )