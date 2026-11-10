import json

base = "dataset/dataset"

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

metadata = load_json(
    f"{base}/metadata/knowledge_base_metadata.json"
)

image_ids = load_json(
    f"{base}/embeddings/image_ids.json"
)

caption_records = load_json(
    f"{base}/embeddings/caption_records.json"
)

# 1. Verify image ID alignment
metadata_ids = [int(x["image_id"]) for x in metadata]
embedding_ids = [int(x) for x in image_ids]

print("========== IMAGE ALIGNMENT ==========")
print("Same number:", len(metadata_ids) == len(embedding_ids))
print("Same order:", metadata_ids == embedding_ids)
print("Unique embedding IDs:", len(set(embedding_ids)))
print("Duplicate IDs:", len(embedding_ids) - len(set(embedding_ids)))

# 2. Verify caption alignment
expected_captions = [
    {
        "image_id": int(item["image_id"]),
        "caption": caption
    }
    for item in metadata
    for caption in item["captions"]
]

actual_captions = [
    {
        "image_id": int(item["image_id"]),
        "caption": item["caption"]
    }
    for item in caption_records
]

print("\n========== CAPTION ALIGNMENT ==========")
print("Expected captions:", len(expected_captions))
print("Actual records:", len(actual_captions))
print("Same order and content:", expected_captions == actual_captions)

print(
    "Duplicate caption records:",
    len(actual_captions) -
    len({
        (x["image_id"], x["caption"])
        for x in actual_captions
    })
)

# 3. Count captions per image
from collections import Counter

counts = Counter(x["image_id"] for x in actual_captions)

print("\n========== CAPTIONS PER IMAGE ==========")
print("Images with exactly 5 captions:",
      sum(n == 5 for n in counts.values()))
print("Images with other counts:",
      sum(n != 5 for n in counts.values()))