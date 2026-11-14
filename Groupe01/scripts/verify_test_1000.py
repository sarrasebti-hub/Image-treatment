from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_common import EXPERIMENT_ROOT, read_json, sha256_ids, split_ids


DIMENSIONS = {"clip-b32": 512, "siglip-b16": 768, "dinov2-s": 384}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    original_ids = split_ids(ROOT / "artifacts" / "splits" / "test.json")
    new_split = read_json(EXPERIMENT_ROOT / "splits" / "test_1000.json")
    new_ids = {int(row["image_id"]) for row in new_split["images"]}

    assert len(original_ids) == 100
    assert len(new_ids) == 1000
    assert not (original_ids & new_ids)
    assert all(row["source"] == "val2017" for row in new_split["images"])

    references = read_jsonl(EXPERIMENT_ROOT / "tables" / "eval_refs_test_1000.jsonl")
    reference_ids = {int(row["image_id"]) for row in references}
    assert reference_ids == new_ids
    counts = {image_id: sum(int(row["image_id"]) == image_id for row in references) for image_id in new_ids}
    assert all(count == 5 for count in counts.values())

    for encoder, dimension in DIMENSIONS.items():
        directory = EXPERIMENT_ROOT / "embeddings" / encoder
        ids = np.load(directory / "ids_test_1000.npy")
        vectors = np.load(directory / "test_1000_img.npy")
        assert ids.shape == (1000,)
        assert vectors.shape == (1000, dimension)
        assert vectors.dtype == np.float32
        assert np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        assert set(int(image_id) for image_id in ids) == new_ids

    frozen_paths = [
        ROOT / "artifacts" / "splits" / "test.json",
        ROOT / "artifacts" / "splits" / "val.json",
        ROOT / "artifacts" / "splits" / "kb.json",
        ROOT / "reports" / "tables" / "encoder_comparison_test.csv",
    ]
    report = {
        "original_test_count": len(original_ids),
        "new_test_count": len(new_ids),
        "overlap_count": len(original_ids & new_ids),
        "new_test_id_hash": sha256_ids(new_ids),
        "frozen_file_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_paths if path.exists()},
    }
    (EXPERIMENT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / "manifests" / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Verified: 100/1000 splits are disjoint and all generated files are aligned")


if __name__ == "__main__":
    main()
