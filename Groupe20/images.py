import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ANN_DIR = Path(r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations")
SPLITS = Path("splits")
IMG_DIR = Path("images")


def url_map(ann_file):
    """image id -> download link"""
    data = json.load(open(ANN_DIR / ann_file))
    return {img["id"]: img["coco_url"] for img in data["images"]}


train_urls = url_map("captions_train2017.json")
val_urls = url_map("captions_val2017.json")

# (ids file, where the links come from, folder to save into)
jobs = [
    ("kb_train_ids.json", train_urls, "kb"),
    ("val_ids.json", train_urls, "val"),
    ("test_ids.json", val_urls, "test"),
    ("test_ids_1000.json", val_urls, "test"),
]

tasks = []
for ids_file, urls, folder in jobs:
    out = IMG_DIR / folder
    out.mkdir(parents=True, exist_ok=True)
    for i in json.load(open(SPLITS / ids_file)):
        tasks.append((urls[i], out / f"{i:012d}.jpg"))


def download(task):
    url, path = task
    if path.exists():            # already downloaded, skip
        return True
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
        return True
    except Exception:
        return False


print("Images to download:", len(tasks))
with ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(download, tasks))

failed = [t for t, ok in zip(tasks, results) if not ok]
print("Done. Failed:", len(failed))