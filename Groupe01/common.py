"""Utilitaires partagés : config, chemins, I/O JSONL, hash, logging, seeds."""
from __future__ import annotations
import hashlib, json, logging, os, platform, random, re, sys, time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict:
    # RAG_CONFIG=config_test.yaml permet de lancer tout le pipeline sur le mini-COCO synthétique
    path = path or os.environ.get("RAG_CONFIG") or ROOT / "config.yaml"
    with open(ROOT / path if not Path(path).is_absolute() else path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def paths(cfg: dict) -> dict[str, Path]:
    a = ROOT / cfg["paths"]["artifacts"]
    p = {
        "raw": ROOT / cfg["paths"]["raw"],
        "art": a,
        "splits": a / "splits",
        "images": a / "images_rgb",
        "tables": a / "tables",
        "emb": a / "embeddings",
        "index": a / "index",
        "eda": a / "eda",
        "qc": a / "qc",
        "audit": a / "audit",
    }
    return p


def mkdirs(p: dict[str, Path]) -> None:
    for k, v in p.items():
        if k != "raw":
            v.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    return logging.getLogger(name)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ["numpy", "PIL", "faiss", "torch", "transformers", "sentence_transformers", "sklearn", "scipy",
              "pandas", "matplotlib"]:
        try:
            mod = __import__(m)
            info[m] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return info


# ---- nettoyage de légendes (UNIQUEMENT pour l'indexation/retrieval ; les références d'évaluation restent brutes)
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']")


def clean_caption(text: str) -> str:
    t = text.strip().lower()
    t = _PUNCT.sub(" ", t)
    t = t.replace(" ' ", " ")
    return _WS.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    return clean_caption(text).split()


# ---- taxonomie de contexte (supercatégories COCO)
INDOOR_SUPERCATS = {"furniture", "appliance", "kitchen", "electronic", "indoor"}
OUTDOOR_SUPERCATS = {"vehicle", "outdoor", "animal"}
INTERACTION_SUPERCATS_NON_PERSON = {"animal", "sports", "vehicle", "kitchen", "food", "furniture", "accessory", "indoor",
                                    "electronic", "appliance", "outdoor"}

COMPLEXITY_BINS = [("simple", 1, 2), ("medium", 3, 5), ("complex", 6, 9), ("crowded", 10, 10 ** 6)]


def complexity_label(n_objects: int) -> str:
    for name, lo, hi in COMPLEXITY_BINS:
        if lo <= n_objects <= hi:
            return name
    return "none"


def context_label(supercats: set[str] | list[str]) -> str:
    s = set(supercats)
    i, o = len(s & INDOOR_SUPERCATS), len(s & OUTDOOR_SUPERCATS)
    if i > o:
        return "indoor"
    if o > i:
        return "outdoor"
    return "mixed"


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
