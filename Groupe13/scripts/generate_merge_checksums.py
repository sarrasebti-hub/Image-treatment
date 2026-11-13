
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "coco_metadata": ROOT / "data/metadata/kb_metadata.json",
    "yolo_detections": ROOT / "data/visual_detections/full_10000/yolo_10000_detections.json",
    "merged_metadata": ROOT / "data/metadata/kb_metadata_with_yolo.json",
    "merge_report": ROOT / "reports/yolo_metadata_merge_report.json"
}

OUTPUT = ROOT / "reports/yolo_merge_sha256.json"


def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Le fichier existe deja : {OUTPUT}"
        )

    results = {}

    for name, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        print(f"Calcul : {name}")
        results[name] = {
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": calculate_sha256(path)
        }

    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\n========== SHA-256 TERMINE ==========")
    print("Fichiers verifies :", len(results))
    print("Rapport cree :", OUTPUT)


if __name__ == "__main__":
    main()
