from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "test_1000_val2017"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_ids(path: Path) -> set[int]:
    data = read_json(path)
    return {int(image["image_id"]) for image in data["images"]}


def sha256_ids(ids) -> str:
    payload = json.dumps(sorted(int(image_id) for image_id in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_sample(ids, count: int, seed: int) -> list[int]:
    ordered = sorted(int(image_id) for image_id in ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:count]

