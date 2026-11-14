"""Quality-control and anti-leakage report for the OFFICIAL 1,000-image test set.

Checks that only need files stored in Git always run. Checks that need the (not versioned)
embedding arrays run when the arrays are present and are reported as SKIP otherwise, so the
report never claims a verification that did not happen.

Usage:  python scripts/qc_test_1000.py
Output: experiments/test_1000_val2017/qc/qc_report_test_1000.{json,md}
Exit code is 1 if any check FAILS.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_common import EXPERIMENT_ROOT, read_json, sha256_ids  # noqa: E402

DIMENSIONS = {"clip-b32": 512, "siglip-b16": 768, "dinov2-s": 384}
NEAR_DUP_COS = 0.97
MIN_SIDE = 200

checks: list[dict] = []


def add(name: str, status: str, detail: str = "") -> None:
    checks.append({"check": name, "status": status, "detail": detail})


def ok(name: str, cond: bool, detail: str = "") -> None:
    add(name, "PASS" if cond else "FAIL", detail)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ids_of(path: Path) -> set[int]:
    return {int(x["image_id"]) for x in read_json(path)["images"]}


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


def sha256_variants(path: Path) -> set[str]:
    """SHA-256 of the raw bytes and of the CRLF form (the release hashes were taken on Windows)."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return {hashlib.sha256(raw).hexdigest(), hashlib.sha256(lf).hexdigest(),
            hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}


def main() -> int:
    split = read_json(EXPERIMENT_ROOT / "splits" / "test_1000.json")
    rows = split["images"]
    test_ids = {int(r["image_id"]) for r in rows}
    kb_ids = ids_of(ROOT / "artifacts/splits/kb.json")
    val_ids = ids_of(ROOT / "artifacts/splits/val.json")
    legacy_ids = ids_of(ROOT / "artifacts/splits/test.json")

    # --- split integrity -------------------------------------------------------------
    ok("test split size = 1000", len(rows) == 1000, f"{len(rows)} rows")
    ok("test image IDs are unique", len(test_ids) == len(rows))
    ok("all test images come from COCO val2017", all(r["source"] == "val2017" for r in rows))
    sm = read_json(EXPERIMENT_ROOT / "manifests" / "split_manifest.json")
    ok("test ID hash matches split_manifest", sha256_ids(test_ids) == sm["selected_id_hash"],
       sha256_ids(test_ids)[:16] + "...")
    ok("minimum image side >= %d px" % MIN_SIDE, all(min(r["width"], r["height"]) >= MIN_SIDE for r in rows))
    ok("every test image has >= 1 object annotation", all(r["n_objects"] >= 1 for r in rows))
    extra = sum(1 for r in rows if r["n_captions"] > 5)
    ok("every test image has >= 5 COCO captions (first 5 kept as references)", all(r["n_captions"] >= 5 for r in rows),
       f"{extra} image(s) have more than 5 captions in COCO; only the first 5 are used")
    ok("KB size = 10000 and validation size = 100", len(kb_ids) == 10000 and len(val_ids) == 100)

    # --- anti-leakage (IDs) ------------------------------------------------------------
    ok("ANTI-LEAK: test ∩ KB = ∅", not (test_ids & kb_ids), f"{len(test_ids & kb_ids)} shared")
    ok("ANTI-LEAK: test ∩ validation = ∅", not (test_ids & val_ids), f"{len(test_ids & val_ids)} shared")
    ok("test ∩ legacy 100-image test = ∅", not (test_ids & legacy_ids), f"{len(test_ids & legacy_ids)} shared")

    # --- references and KB captions -----------------------------------------------------
    refs = read_jsonl(EXPERIMENT_ROOT / "tables" / "eval_refs_test_1000.jsonl")
    kb_caps = read_jsonl(ROOT / "artifacts/tables/kb_captions.jsonl")
    per_image = Counter(int(r["image_id"]) for r in refs)
    ok("5000 reference rows", len(refs) == 5000, f"{len(refs)} rows")
    ok("references cover exactly the test images", set(per_image) == test_ids)
    ok("5 references per test image", set(per_image.values()) == {5}, f"min={min(per_image.values())} max={max(per_image.values())}")
    ok("reference caption_id are unique", len({r["caption_id"] for r in refs}) == len(refs))
    ok("no empty reference caption", all(str(r["caption_raw"]).strip() for r in refs))
    kb_cap_images = {int(r["image_id"]) for r in kb_caps}
    kb_cap_ids = {r["caption_id"] for r in kb_caps}
    ok("ANTI-LEAK: no test image has captions in the KB", not (test_ids & kb_cap_images))
    ok("ANTI-LEAK: no test caption_id in the KB", not ({r["caption_id"] for r in refs} & kb_cap_ids))
    kb_norm = {norm(r["caption_raw"]) for r in kb_caps}
    matches = [r for r in refs if norm(r["caption_raw"]) in kb_norm]
    add("info: test references textually identical to a KB caption", "PASS",
        f"{len(matches)}/{len(refs)} rows ({100 * len(matches) / len(refs):.1f}%) on {len({r['image_id'] for r in matches})} images; "
        "generic captions of DIFFERENT images, not same-image leakage")

    # --- metadata --------------------------------------------------------------------------
    meta = read_jsonl(EXPERIMENT_ROOT / "tables" / "metadata_test_1000.jsonl")
    ok("metadata exists for every test image", {int(m["image_id"]) for m in meta} == test_ids, f"{len(meta)} rows")
    ok("metadata image IDs are unique", len({m["image_id"] for m in meta}) == len(meta))
    classes = {o["category"] for m in meta for o in m["objects"]}
    add("info: object classes covered by the test set", "PASS", f"{len(classes)}/80")

    # --- frozen legacy release untouched ----------------------------------------------------
    ver = read_json(EXPERIMENT_ROOT / "manifests" / "verification.json")
    for rel, expected in ver["frozen_file_hashes"].items():
        path = ROOT / rel.replace("\\", "/")
        ok(f"frozen file unchanged: {rel.replace(chr(92), '/')}", path.exists() and expected in sha256_variants(path),
           "content identical (hash compared modulo LF/CRLF line endings)")

    # --- results table -------------------------------------------------------------------------
    res_path = EXPERIMENT_ROOT / "results" / "encoder_comparison_test_1000.csv"
    with res_path.open(encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    ok("results cover 5 configurations x 1000 queries",
       len(results) == 5 and all(int(r["n_queries"]) == 1000 for r in results))
    ok("every configuration beats the random-caption baseline (CIDEr)",
       all(float(r["cider"]) > float(r["random_caption_cider_seed42"]) for r in results),
       "; ".join(f"{r['config']}={float(r['cider']):.3f}" for r in results))

    # --- embeddings (only if present locally) ---------------------------------------------------
    try:
        import numpy as np
    except ImportError:  # numpy is a hard requirement of the project, but stay robust
        np = None
    for enc, dim in DIMENSIONS.items():
        d = EXPERIMENT_ROOT / "embeddings" / enc
        ids_p, vec_p = d / "ids_test_1000.npy", d / "test_1000_img.npy"
        if np is None or not (ids_p.exists() and vec_p.exists()):
            add(f"embeddings {enc}/test_1000", "SKIP", "arrays not present locally (not versioned in Git)")
            continue
        ids, vec = np.load(ids_p), np.load(vec_p)
        ok(f"embeddings {enc}/test_1000 shape/dtype", vec.shape == (1000, dim) and vec.dtype == np.float32, f"shape={vec.shape}")
        ok(f"embeddings {enc}/test_1000 finite and L2-normalised",
           bool(np.isfinite(vec).all()) and bool(np.allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-5)))
        ok(f"embeddings {enc}/test_1000 IDs match split", {int(i) for i in ids} == test_ids)
        kb_vec_p, kb_ids_p = ROOT / "artifacts/embeddings" / enc / "kb_img.npy", ROOT / "artifacts/embeddings" / enc / "ids_kb.npy"
        if kb_vec_p.exists() and kb_ids_p.exists():
            kb_vec, kb_ids_arr = np.load(kb_vec_p), np.load(kb_ids_p)
            sims = vec @ kb_vec.T
            top = sims.argmax(axis=1)
            ok(f"ANTI-LEAK {enc}: no test image retrieved as itself in KB",
               not ({int(i) for i in kb_ids_arr[top]} & test_ids))
            near = int((sims.max(axis=1) >= NEAR_DUP_COS).sum())
            add(f"info {enc}: test images with a KB near-duplicate (cos >= {NEAR_DUP_COS})", "PASS", f"{near}/1000")
        else:
            add(f"embeddings {enc}: KB-based leakage checks", "SKIP", "KB embeddings not present locally")

    # --- write report ---------------------------------------------------------------------------
    n = Counter(c["status"] for c in checks)
    summary = {"release": "g1-v1.2", "dataset": "test_1000_val2017", "n_checks": len(checks),
               "pass": n["PASS"], "fail": n["FAIL"], "skip": n["SKIP"], "checks": checks}
    out = EXPERIMENT_ROOT / "qc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc_report_test_1000.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    verdict = "PASS" if n["FAIL"] == 0 else "FAIL"
    lines = [f"# QC — official test set (1,000 images, COCO val2017), release g1-v1.2", "",
             f"**{verdict}** — {len(checks)} checks: {n['PASS']} PASS, {n['FAIL']} FAIL, {n['SKIP']} SKIP "
             "(SKIP = needs arrays that are not versioned in Git; rerun locally after generating embeddings).", "",
             "| # | Status | Check | Detail |", "|---:|---|---|---|"]
    for i, c in enumerate(checks, 1):
        lines.append(f"| {i} | {c['status']} | {c['check']} | {c['detail'].replace('|', '/')} |")
    (out / "qc_report_test_1000.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{verdict}: {n['PASS']} pass, {n['FAIL']} fail, {n['SKIP']} skip")
    for c in checks:
        if c["status"] == "FAIL":
            print("  FAIL:", c["check"], c["detail"])
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
