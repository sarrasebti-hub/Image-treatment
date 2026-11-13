
import json
from pathlib import Path
from collections import Counter

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

COCO_METADATA_FILE = ROOT / "data/metadata/kb_metadata.json"

YOLO_DETECTIONS_FILE = (
    ROOT / "data/visual_detections/full_10000/yolo_10000_detections.json"
)

OUTPUT_FILE = ROOT / "data/metadata/kb_metadata_with_yolo.json"


# ==================================================
# 2. CHARGEMENT DES DONNEES
# ==================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==================================================
# 3. VERIFICATION DES IDENTIFIANTS
# ==================================================

def build_index(records, source_name):
    index = {}

    for record in records:
        image_id = record.get("image_id")

        if image_id is None:
            raise ValueError(
                f"image_id manquant dans {source_name}"
            )

        if image_id in index:
            raise ValueError(
                f"image_id duplique dans {source_name} : {image_id}"
            )

        index[image_id] = record

    return index


# ==================================================
# 4. FUSION DES METADONNEES
# ==================================================

def merge_metadata():

    print("Chargement des fichiers...")

    coco_records = load_json(COCO_METADATA_FILE)
    yolo_records = load_json(YOLO_DETECTIONS_FILE)

    if not isinstance(coco_records, list):
        raise TypeError("Les metadonnees COCO doivent etre une liste.")

    if not isinstance(yolo_records, list):
        raise TypeError("Les detections YOLO doivent etre une liste.")

    print("Metadonnees COCO :", len(coco_records))
    print("Enregistrements YOLO :", len(yolo_records))

    coco_index = build_index(coco_records, "COCO")
    yolo_index = build_index(yolo_records, "YOLO")

    coco_ids = set(coco_index)
    yolo_ids = set(yolo_index)

    missing_ids = coco_ids - yolo_ids
    extra_ids = yolo_ids - coco_ids

    if missing_ids or extra_ids:
        raise ValueError(
            f"Identifiants incompatibles : "
            f"{len(missing_ids)} manquants, "
            f"{len(extra_ids)} supplementaires."
        )

    # Ne pas modifier les donnees COCO originales.
    merged_records = []

    total_detections = 0
    category_counter = Counter()

    for coco_record in coco_records:

        image_id = coco_record["image_id"]
        yolo_record = yolo_index[image_id]

        detections = yolo_record.get("detections")

        if not isinstance(detections, list):
            raise TypeError(
                f"Format de detections invalide pour image {image_id}"
            )

        categories = sorted({
            detection["class_name"]
            for detection in detections
        })

        # Copie du dictionnaire COCO.
        merged_record = dict(coco_record)

        # Ajout des informations YOLO.
        merged_record["yolo_detections"] = detections
        merged_record["yolo_detection_count"] = len(detections)
        merged_record["yolo_detected_categories"] = categories

        merged_records.append(merged_record)

        total_detections += len(detections)
        category_counter.update(
            detection["class_name"]
            for detection in detections
        )

    # ==================================================
    # 5. SAUVEGARDE
    # ==================================================

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Le fichier de sortie existe deja : {OUTPUT_FILE}\n"
            "Aucune donnee n'a ete ecrasee."
        )

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    try:
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(
                merged_records,
                file,
                ensure_ascii=False,
                indent=2
            )

        temporary_file.replace(OUTPUT_FILE)

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise

    # ==================================================
    # 6. RAPPORT
    # ==================================================

    print("\n========== FUSION TERMINEE ==========")
    print("Images fusionnees :", len(merged_records))
    print("Total detections YOLO :", total_detections)
    print("Categories YOLO :", len(category_counter))
    print("Images sans detection :", sum(
        1 for record in merged_records
        if record["yolo_detection_count"] == 0
    ))
    print("Fichier cree :", OUTPUT_FILE)


# ==================================================
# 7. EXECUTION
# ==================================================

if __name__ == "__main__":
    merge_metadata()
