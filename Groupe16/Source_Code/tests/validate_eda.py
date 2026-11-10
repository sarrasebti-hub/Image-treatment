"""Manual, offline EDA/render validation on SYNTHETIC data only.

Run after `python -m src.cli download-nlp`:
    python -m tests.validate_eda
Output is confined to tmp/synthetic_eda; never feeds the phase-1 scientific report.
"""
from pathlib import Path

import nbformat
import nltk

from src.data.coco import CocoData
from src.data.splits import create_splits
from src.exploration.eda import run_eda
from src.preprocessing.metadata import prepare_metadata
from src.utils.core import load_config, path, write_json
from tests.test_phase1 import fixture


if __name__ == "__main__":
    original = load_config()
    nltk.data.path.insert(0, str(path(original, "cache") / "nltk_data"))
    root = Path("tmp/synthetic_eda").resolve()
    cfg = fixture(root)
    data = CocoData(cfg)
    create_splits(cfg, data, force=True)
    prepare_metadata(cfg, data)
    results = run_eda(cfg, data)
    assert results["splits"]["knowledge_base"]["images"] == cfg["kb_size"]
    assert results["splits"]["knowledge_base"]["captions"] == cfg["kb_size"] * cfg["expected_captions"]
    assert results["splits"]["knowledge_base"]["vocabulary_size"] > 0
    for filename in Path("notebooks").glob("*.ipynb"):
        nbformat.validate(nbformat.read(filename, as_version=4))
    write_json(root / "validation.json", {"status": "passed", "purpose": "Synthetic integration/render check only", "splits": list(results["splits"])})
    print("Synthetic EDA and notebook schema validation passed; outputs in tmp/synthetic_eda")
