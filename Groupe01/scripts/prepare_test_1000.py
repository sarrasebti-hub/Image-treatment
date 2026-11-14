from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_common import EXPERIMENT_ROOT, read_json


def fetch(url: str, destination: Path, attempts: int = 4) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True
    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(url, destination)
            return True
        except Exception as exc:
            print(f"Attempt {attempt + 1} failed for {url}: {exc}")
            time.sleep(1.5 * (attempt + 1))
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    split = read_json(EXPERIMENT_ROOT / "splits" / "test_1000.json")
    raw_dir = args.raw / "images" / "val2017"
    output_dir = EXPERIMENT_ROOT / "images_rgb" / "test_1000"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(row["coco_url"], raw_dir / row["file_name"]) for row in split["images"]]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
        ok = sum(executor.map(lambda item: fetch(item[0], item[1]), tasks))
    if ok != len(tasks):
        raise RuntimeError(f"Downloaded {ok}/{len(tasks)} images")

    for row in split["images"]:
        source = raw_dir / row["file_name"]
        destination = output_dir / f"{int(row['image_id']):012d}.jpg"
        with Image.open(source).convert("RGB") as image:
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            image.save(destination, format="JPEG", quality=args.jpeg_quality, optimize=True)

    print(f"Prepared {len(split['images'])} preprocessed images in {output_dir}")


if __name__ == "__main__":
    main()
