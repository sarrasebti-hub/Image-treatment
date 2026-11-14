from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocoevalcap.cider.cider import Cider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import clean_caption
from scripts.experiment_common import EXPERIMENT_ROOT


def load_module():
    path = ROOT / "scripts" / "compare_encoders.py"
    spec = importlib.util.spec_from_file_location("compare_encoders", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    comparison = load_module()
    artifacts = ROOT / "artifacts"
    frozen_metadata = {
        int(row["image_id"]): row
        for row in read_jsonl(artifacts / "tables" / "metadata.jsonl")
    }
    new_metadata = {
        int(row["image_id"]): row
        for row in read_jsonl(EXPERIMENT_ROOT / "tables" / "metadata_test_1000.jsonl")
    }
    metadata = {**frozen_metadata, **new_metadata}
    kb_captions = read_jsonl(artifacts / "tables" / "kb_captions.jsonl")

    references = defaultdict(list)
    for row in read_jsonl(EXPERIMENT_ROOT / "tables" / "eval_refs_test_1000.jsonl"):
        references[int(row["image_id"])].append(row["caption_raw"])

    rng = np.random.RandomState(42)
    random_predictions = {
        image_id: kb_captions[int(rng.randint(0, len(kb_captions)))]["caption_raw"]
        for image_id in sorted(references)
    }
    random_scores = comparison.cider_per_image(random_predictions, dict(references))
    random_cider = float(np.mean(random_scores))

    rows = []
    for encoder, mode in comparison.CONFIGS:
        embedding_dir = EXPERIMENT_ROOT / "embeddings" / encoder
        query_ids = np.load(embedding_dir / "ids_test_1000.npy")
        query_vectors = np.load(embedding_dir / "test_1000_img.npy")
        if set(int(image_id) for image_id in query_ids) != set(references):
            raise ValueError(f"{encoder}: query IDs do not match references")

        retrieved_classes, predictions_by_row, latency_ms = comparison.retrieve(
            encoder,
            mode,
            np.ascontiguousarray(query_vectors, dtype=np.float32),
            10,
            kb_captions,
            metadata,
        )
        predictions = {
            int(query_ids[row_idx]): caption
            for row_idx, caption in predictions_by_row.items()
        }
        if set(predictions) != set(references):
            raise ValueError(f"{encoder}|{mode}: incomplete predictions")

        ids = sorted(references)
        _, cider_scores = Cider().compute_score(
            {image_id: [clean_caption(text) for text in references[image_id]] for image_id in ids},
            {image_id: [clean_caption(predictions[image_id])] for image_id in ids},
        )
        cider = float(np.mean(cider_scores))
        no_objects, metrics = comparison.relevance_metrics(query_ids, retrieved_classes, metadata)
        rows.append({
            "dataset": "test_1000",
            "experiment": "test_1000_val2017",
            "source": "val2017",
            "report_note": "independent second test, disjoint from test_100",
            "config": f"{encoder}|{mode}",
            "encoder": encoder,
            "mode": mode,
            "n_queries": len(query_ids),
            "queries_without_objects": no_objects,
            "relevance_queries": len(query_ids) - no_objects,
            **metrics,
            "cider": cider,
            "random_caption_cider_seed42": random_cider,
            "latency_ms_per_query_median_5_repeats": latency_ms,
        })

    output_dir = EXPERIMENT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "encoder_comparison_test_1000.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "encoder_comparison_test_1000.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    comparison.write_figure(rows, output_dir / "encoder_comparison_test_1000.png")
    print(f"Created {csv_path}")
    print(f"Created {json_path}")


if __name__ == "__main__":
    main()
