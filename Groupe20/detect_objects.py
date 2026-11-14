import json
import time
from pathlib import Path

import ultralytics
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"   
CONF_MIN = 0.25             
IMGSZ = 640
CHUNK = 64

OUT = Path("metadata")
OUT.mkdir(exist_ok=True)

jobs = [
    ("kb_train_ids.json", "kb", "detected_kb.json"),
    ("val_ids.json", "val", "detected_val.json"),
    ("test_ids_1000.json", "test", "detected_test1000.json"),
]


missing = []
for ids_file, folder, _ in jobs:
    for i in json.load(open(f"splits/{ids_file}")):
        if not Path(f"images/{folder}/{i:012d}.jpg").exists():
            missing.append((folder, i))
if missing:
    raise SystemExit(f"{len(missing)} images are missing (first: {missing[0]}). Run images.py first.")

model = YOLO(MODEL_NAME)

for ids_file, folder, out_name in jobs:
    out_path = OUT / out_name
    if out_path.exists():
        print("skip (already done):", out_name)
        continue

    ids = json.load(open(f"splits/{ids_file}"))
    images = {}
    t0 = time.time()
    for s in range(0, len(ids), CHUNK):
        chunk = ids[s:s + CHUNK]
        paths = [f"images/{folder}/{i:012d}.jpg" for i in chunk]
        results = model.predict(paths, conf=CONF_MIN, imgsz=IMGSZ, device="cpu", verbose=False)
        for i, r in zip(chunk, results):
            objs = []
            for (x1, y1, x2, y2), c, p in zip(r.boxes.xyxy.tolist(),
                                               r.boxes.cls.tolist(),
                                               r.boxes.conf.tolist()):
                objs.append({
                    "category": r.names[int(c)],
                    "confidence": round(p, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                })
            images[str(i)] = {"objects": objs}
        if (s // CHUNK) % 10 == 0:
            print(f"{folder}: {s}/{len(ids)}  ({time.time() - t0:.0f}s)")

    result = {
        "detector": MODEL_NAME,
        "ultralytics_version": ultralytics.__version__,
        "min_confidence": CONF_MIN,
        "image_size": IMGSZ,
        "device": "cpu",
        "n_images": len(ids),
        "seconds": round(time.time() - t0, 1),
        "images": images,
    }
    json.dump(result, open(out_path, "w"))

    for thr in (0.25, 0.5):
        counts = [sum(o["confidence"] >= thr for o in v["objects"]) for v in images.values()]
        print(f"{folder} @ conf>={thr}: avg objects {sum(counts) / len(counts):.2f}, "
              f"images with none: {sum(c == 0 for c in counts)}")
    print("saved", out_path)