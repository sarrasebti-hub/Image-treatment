from collections import defaultdict
from pathlib import Path
import random

from src.utils.core import digest, path, read_json, stamp, write_json

SPLIT_SOURCES = {"knowledge_base": "train2017", "validation": "train2017", "test": "val2017"}


def complexity(count, cfg):
    if count <= cfg["analysis"]["simple_max_objects"]:
        return "simple"
    if count <= cfg["analysis"]["medium_max_objects"]:
        return "medium"
    return "complex"


def check_splits(splits):
    for name, ids in splits.items():
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError(f"{name}: expected a JSON list of integer COCO IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate IDs")
    names = list(splits)
    for j, name in enumerate(names):
        for other in names[j + 1:]:
            if set(splits[name]) & set(splits[other]):
                raise ValueError(f"Split overlap: {name}/{other}")


def load_splits(cfg):
    result = {name: read_json(path(cfg, "splits") / f"{name}_ids.json") for name in SPLIT_SOURCES}
    check_splits(result)
    manifest = read_json(path(cfg, "splits") / "split_manifest.json")
    if digest(result) != manifest["ids_sha256"]:
        raise ValueError("Split IDs changed after freezing")
    for name, key in (("knowledge_base", "kb_size"), ("validation", "validation_size"), ("test", "test_size")):
        if len(result[name]) != cfg[key]:
            raise ValueError(f"{name}: split size differs from configuration")
    comparison = manifest.get("comparison_test")
    if comparison:
        legacy = read_json(path(cfg, "splits") / comparison["ids_file"])
        if len(legacy) != comparison["size"] or digest(legacy) != comparison["ids_sha256"]:
            raise ValueError("Original comparison test IDs changed")
        if not set(legacy).issubset(result["test"]):
            raise ValueError("Original comparison test is no longer contained in extended test")
    return result


def extend_test(cfg, data):
    """Grow only the test split; preserve KB, validation and the original test IDs."""
    data.require_valid()
    folder = path(cfg, "splits")
    manifest = read_json(folder / "split_manifest.json")
    current = {s: read_json(folder / f"{s}_ids.json") for s in SPLIT_SOURCES}
    check_splits(current)
    if digest(current) != manifest["ids_sha256"] or data.annotation_hashes != manifest["annotation_sha256"]:
        raise ValueError("Original split or annotation identity changed; refusing migration")
    if len(current["test"]) == cfg["test_size"]:
        return {"status": "already_extended", "sizes": manifest["sizes"]}
    if cfg["test_size"] <= len(current["test"]):
        raise ValueError("extend-test only grows the current test")
    if len(current["knowledge_base"]) != cfg["kb_size"] or len(current["validation"]) != cfg["validation_size"]:
        raise ValueError("KB or validation size mismatch")
    official = cfg["splits"]["official_test_ids"]
    candidate = Path(cfg["_root"]) / (official or cfg["splits"]["official_test_candidate"])
    if candidate.is_file():
        raise ValueError("Common teacher IDs require an explicit migration respecting that list; do not merge or sample them silently")
    eligible = set(data.eligible("val2017"))
    if not set(current["test"]).issubset(eligible):
        raise ValueError("Original test contains ineligible images")
    old_ids = current["test"]
    comparison_file = folder / "comparison/test_100_ids.json"
    if comparison_file.exists() and read_json(comparison_file) != old_ids:
        raise ValueError("Refusing to overwrite another comparison test")
    write_json(comparison_file, old_ids)
    extra = stratified_sample(sorted(eligible - set(old_ids)), cfg["test_size"]-len(old_ids),
                              data, "val2017", cfg, random.Random(cfg["random_seed"]))
    current["test"] = sorted(old_ids + extra)
    check_splits(current)
    write_json(folder / "test_ids.json", current["test"])
    manifest.update({**stamp(cfg), "ids_sha256": digest(current),
                     "sizes": {k: len(v) for k, v in current.items()},
                     "comparison_test": {"ids_file": "comparison/test_100_ids.json", "size": len(old_ids),
                                         "ids_sha256": digest(old_ids), "relationship": "nested subset of test; never a KB/validation split"},
                     "extension": {"added": len(extra), "kb_and_validation_ids_unchanged": True,
                                   "method": "Retain first 100; largest-remainder stratified sample of 900 remaining eligible val2017 images, seed 42"},
                     "specification_ambiguity": "Resolved by teacher clarification relayed 7 November 2026: test target 1000; retain original 100"})
    write_json(folder / "split_manifest.json", manifest)
    return manifest


def stratified_sample(ids, n, data, source, cfg, rng):
    if n > len(ids):
        raise ValueError(f"Need {n} eligible {source} images, found {len(ids)}")
    strata = defaultdict(list)
    for i in sorted(ids):
        strata[complexity(len(data.objects(source, i)), cfg)].append(i)
    quotas = {s: n * len(v) / len(ids) for s, v in strata.items()}
    counts = {s: int(q) for s, q in quotas.items()}
    order = sorted(strata, key=lambda s: (-(quotas[s] - counts[s]), s))
    for s in order[:n - sum(counts.values())]:
        counts[s] += 1
    chosen = []
    for s in sorted(strata):
        chosen.extend(rng.sample(strata[s], counts[s]))
    return sorted(chosen)


def create_splits(cfg, data, force=False):
    data.require_valid()
    destination = path(cfg, "splits")
    existing = list(destination.glob("*_ids.json"))
    if existing and not force:
        load_splits(cfg)
        raise FileExistsError("Frozen splits exist. Reuse them; --force explicitly replaces the local split release.")
    official = cfg["splits"]["official_test_ids"]
    candidate = Path(cfg["_root"]) / cfg["splits"]["official_test_candidate"]
    official_path = Path(cfg["_root"]) / official if official else candidate
    if official and not official_path.is_file():
        raise FileNotFoundError(official_path)
    rng = random.Random(cfg["random_seed"])
    train = data.eligible("train2017")
    val = data.eligible("val2017")
    is_official = official_path.is_file()
    if is_official:
        test = read_json(official_path)
        check_splits({"test": test})
        if len(test) != cfg["test_size"] or not set(test).issubset(val):
            raise ValueError("Official test IDs incompatible with test_size or annotations. Do not subsample teacher IDs.")
    else:
        test = stratified_sample(val, cfg["test_size"], data, "val2017", cfg, rng)
    train = sorted(set(train) - set(test))
    validation = stratified_sample(train, cfg["validation_size"], data, "train2017", cfg, rng)
    kb = stratified_sample(sorted(set(train) - set(validation)), cfg["kb_size"], data, "train2017", cfg, rng)
    splits = {"knowledge_base": kb, "validation": validation, "test": test}
    check_splits(splits)
    for name, ids in splits.items():
        write_json(destination / f"{name}_ids.json", ids)
    exclusions = {source: sorted(set(data.captions[source].imgs) - set(data.eligible(source))) for source in data.captions}
    write_json(destination / "sampling_exclusions.json", {
        "reason": "Missing instance-image record or caption count differs from expected_captions; not silently removed",
        "ids_by_source": exclusions,
    })
    manifest = {
        **stamp(cfg), "status": "teacher_supplied" if is_official else "PROVISIONAL_NOT_TEACHER_SPLIT",
        "official_ids_file": str(official_path.relative_to(Path(cfg["_root"]))) if is_official and official_path.is_relative_to(Path(cfg["_root"])) else (str(official_path) if is_official else None),
        "ids_sha256": digest(splits), "annotation_sha256": data.annotation_hashes,
        "sources": SPLIT_SOURCES, "sizes": {k: len(v) for k, v in splits.items()},
        "seed": cfg["random_seed"], "sampling": "Largest-remainder proportional allocation by derived instance-count complexity; seeded random sample within strata",
        "limits": "Complexity stratification does not guarantee interactions/context coverage. Teacher common IDs supersede provisional test.",
        "complexity_thresholds": {k: cfg["analysis"][k] for k in ("simple_max_objects", "medium_max_objects")},
        "specification_ambiguity": "100 test images in sections 4.1-4.2; 1000 in 4.4. Teacher confirmation pending.",
    }
    write_json(destination / "split_manifest.json", manifest)
    return manifest
