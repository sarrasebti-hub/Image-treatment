import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def load_config(filename="configs/config.yaml"):
    filename = Path(filename).resolve()
    cfg = yaml.safe_load(filename.read_text(encoding="utf-8"))
    cfg["_root"] = str(filename.parent.parent)
    for key in ("kb_size", "validation_size", "test_size", "expected_captions"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["embedding"]["normalize"] is not True:
        raise ValueError("Cosine IndexFlatIP requires normalize: true")
    if cfg["embedding"]["dtype"] != "float32":
        raise ValueError("This phase-1 adapter exports float32 only")
    if cfg["analysis"]["pos_resource"] != "averaged_perceptron_tagger_eng":
        raise ValueError("English captions require the configured English POS tagger")
    if cfg["splits"]["strategy"] != "complexity_stratified":
        raise ValueError("Unsupported sampling strategy")
    if cfg["analysis"]["simple_max_objects"] >= cfg["analysis"]["medium_max_objects"]:
        raise ValueError("Complexity thresholds must be increasing")
    policy = cfg.get("metadata", {})
    if policy and (policy.get("human_caption_confidence") != 1.0 or policy.get("human_caption_threshold") != 1.0
                   or policy.get("confidence_policy") != "human_reference_convention_only"):
        raise ValueError("Human reference policy requires confidence=threshold=1; it is not a detector policy")
    return cfg


def path(cfg, key):
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def read_json(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def write_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers/watchers can briefly omit FILE_SHARE_DELETE while opening JSON.
    # Keep the previous complete file visible and retry the atomic replacement.
    for attempt in range(8):
        try:
            temporary.replace(filename)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.0))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(cfg):
    packages = ("numpy", "Pillow", "nltk", "pycocotools", "torch", "transformers", "faiss-cpu", "PyYAML", "matplotlib", "huggingface-hub", "tqdm")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": cfg["preprocessing"]["version"],
        "config_sha256": digest({k: v for k, v in cfg.items() if not k.startswith("_")}),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite 2D embedding matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero embedding")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)
