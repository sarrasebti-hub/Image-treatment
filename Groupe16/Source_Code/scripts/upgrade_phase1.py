"""Explicit two-stage upgrade; preserve previous release and reuse verified KB vectors.

backup: before extend-test/download-images/metadata.
reuse: after metadata, before index/EDA/validation/QC.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.core import load_config, read_json, write_json, file_hash, digest
from src.embeddings.encode import load_embeddings, encoder_inputs, input_identity
from src.preprocessing.metadata import load_metadata

BACKUP = ROOT / "reports/releases/phase1_test100_before_extension"


def backup():
    if BACKUP.exists():
        raise FileExistsError("Previous release archive already exists; refusing replacement")
    old = load_config(ROOT / "reports/config_used.yaml")
    rows = [json.loads(line) for line in (ROOT / "artifacts/metadata/knowledge_base.jsonl").read_text(encoding="utf-8").splitlines()]
    arrays, maps, em = load_embeddings(old, rows)
    names = ["reports/config_used.yaml", "reports/eda.json", "reports/audit.json", "reports/quality_control.json",
             "reports/run_status.json", "reports/retrieval/validation/results.json", "artifacts/metadata/manifest.json",
             "artifacts/embeddings/manifest.json", "artifacts/indexes/manifest.json"]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "artifacts/splits").glob("*.json")]
    names += [f"artifacts/metadata/{s}.jsonl" for s in ("knowledge_base", "validation", "test")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "output/pdf").glob("*") if p.is_file()]
    for name in names:
        destination = BACKUP / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    evidence = {"archived_utc": datetime.now(timezone.utc).isoformat(),
                "original_files_sha256": {name: file_hash(ROOT / name) for name in names},
                "kb_encoder_inputs_sha256": digest(encoder_inputs(rows)),
                "image_npy_sha256": file_hash(ROOT / "artifacts/embeddings/image.npy"),
                "text_npy_sha256": file_hash(ROOT / "artifacts/embeddings/text.npy"),
                "policy": "Archive original test100 release. No embedding/image copies or sampling changes in this stage."}
    write_json(BACKUP / "archive_manifest.json", evidence)
    print(json.dumps({"backup": str(BACKUP), "kb_vectors_verified": len(arrays["image"])}, indent=2))


def reuse():
    cfg = load_config()
    evidence = read_json(BACKUP / "archive_manifest.json")
    rows = load_metadata(cfg, "knowledge_base")
    old_rows = [json.loads(line) for line in (BACKUP / "artifacts/metadata/knowledge_base.jsonl").read_text(encoding="utf-8").splitlines()]
    old_cfg = load_config(BACKUP / "reports/config_used.yaml")
    old_cfg["_root"] = str(ROOT)
    arrays, maps, old_manifest = load_embeddings(old_cfg, old_rows)
    if encoder_inputs(rows) != encoder_inputs(old_rows):
        raise ValueError("Actual encoder inputs changed; cached vectors cannot be reused")
    for s in ("knowledge_base", "validation"):
        if file_hash(ROOT / f"artifacts/splits/{s}_ids.json") != evidence["original_files_sha256"][f"artifacts/splits/{s}_ids.json"]:
            raise ValueError("KB/validation IDs changed")
    for row in rows:
        if file_hash(ROOT / "data/raw" / row["coco_source"] / row["file_name"]) != row["file_sha256"]:
            raise ValueError("KB original image changed")
    manifest = dict(old_manifest)
    manifest.update({"identity_schema_version": 2, "input_sha256": input_identity(cfg, rows),
                     "config_sha256": digest({k:v for k,v in cfg.items() if not k.startswith("_")}),
                     "provenance_updated_utc": datetime.now(timezone.utc).isoformat(),
                     "verified_reuse": {"reason": "Test-only extension and human-reference confidence fields do not change encoder inputs",
                                       "encoder_inputs_identical": True, "kb_vectors_recomputed": False,
                                       "original_manifest_sha256": file_hash(BACKUP / "artifacts/embeddings/manifest.json")}})
    write_json(ROOT / "artifacts/embeddings/manifest.json", manifest)
    load_embeddings(cfg, rows)
    progress_file = ROOT / "artifacts/embeddings/progress.json"
    progress = read_json(progress_file)
    progress.update({"original_checkpoint_input_sha256": progress["input_sha256"],
                     "input_sha256": manifest["input_sha256"], "stage": "complete_reused_verified",
                     "last_verified_utc": manifest["provenance_updated_utc"]})
    write_json(progress_file, progress)
    assert file_hash(ROOT / "artifacts/embeddings/image.npy") == evidence["image_npy_sha256"]
    assert file_hash(ROOT / "artifacts/embeddings/text.npy") == evidence["text_npy_sha256"]
    write_json(ROOT / "reports/phase1_upgrade.json", {"status": "verified_vector_reuse", "old_test_size": 100,
        "new_test_size": cfg["test_size"], "kb_and_validation_ids_unchanged": True,
        "image_vectors_unchanged": True, "text_vectors_unchanged": True, "kb_encoder_inputs_identical": True,
        "kb_embeddings_recomputed": False, "original_release": str(BACKUP.relative_to(ROOT)),
        "verified_utc": datetime.now(timezone.utc).isoformat()})
    print("Verified original vectors reused; no KB inference or training performed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("backup", "reuse")); a=p.parse_args()
    backup() if a.stage == "backup" else reuse()
