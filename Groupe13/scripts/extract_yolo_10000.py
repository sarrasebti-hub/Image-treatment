
import csv
import json
import time
import math
from pathlib import Path
from collections import Counter

from ultralytics import YOLO


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

KB_IDS_FILE = ROOT / "data/splits/knowledge_base_image_ids.json"
IMAGE_DIR = ROOT / "data/train2017"
MODEL_PATH = ROOT / "yolo26n.pt"

OUTPUT_DIR = ROOT / "data/visual_detections/full_10000"

CHECKPOINT_FILE = OUTPUT_DIR / "detections_checkpoint.json"
OUTPUT_JSON = OUTPUT_DIR / "yolo_10000_detections.json"
OUTPUT_CSV = OUTPUT_DIR / "yolo_10000_detections.csv"
STATISTICS_FILE = OUTPUT_DIR / "yolo_10000_statistics.json"

CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cpu"
SAVE_EVERY = 100

# Sécurité : ne pas lancer l'extraction pendant la préparation.
RUN_EXTRACTION = True


# ==================================================
# 2. FONCTIONS DE SAUVEGARDE
# ==================================================

def save_json(data, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def save_csv(detections, path):
    fieldnames = [
        "image_id", "file_name", "class_id", "class_name",
        "confidence", "x1", "y1", "x2", "y2"
    ]

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for image in detections:
            for detection in image["detections"]:
                writer.writerow({
                    "image_id": image["image_id"],
                    "file_name": image["file_name"],
                    **detection
                })

    temporary_path.replace(path)


# ==================================================
# 3. CHARGEMENT ET VALIDATION DU CHECKPOINT
# ==================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():
        return []

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Checkpoint invalide : une liste était attendue."
        )

    completed_ids = set()

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            raise ValueError(
                f"Entrée invalide à la position {index}."
            )

        required_fields = {
            "image_id",
            "file_name",
            "detections"
        }

        if not required_fields.issubset(item):
            raise ValueError(
                f"Champs manquants dans l'entrée {index}."
            )

        image_id = item["image_id"]

        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(
                f"ID invalide à la position {index}."
            )

        if image_id in completed_ids:
            raise ValueError(
                f"ID dupliqué dans le checkpoint : {image_id}"
            )

        completed_ids.add(image_id)

        expected_file_name = f"{image_id:012d}.jpg"

        if item["file_name"] != expected_file_name:
            raise ValueError(
                f"Nom de fichier incorrect pour l'image {image_id}."
            )

        image_detections = item["detections"]

        if not isinstance(image_detections, list):
            raise ValueError(
                f"Les détections de l'image {image_id} doivent être une liste."
            )

        for detection in image_detections:

            if not isinstance(detection, dict):
                raise ValueError(
                    f"Détection invalide pour l'image {image_id}."
                )

            required_detection_fields = {
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2"
            }

            if not required_detection_fields.issubset(detection):
                raise ValueError(
                    f"Champs manquants dans une détection de l'image {image_id}."
                )

            class_id = detection["class_id"]

            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ValueError("class_id invalide.")

            if class_id < 0:
                raise ValueError("class_id négatif.")

            if not isinstance(detection["class_name"], str):
                raise ValueError("class_name invalide.")

            numeric_fields = [
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2"
            ]

            for field in numeric_fields:

                value = detection[field]

                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"Valeur numérique invalide : {field}, "
                        f"image {image_id}."
                    )

            if not 0 <= detection["confidence"] <= 1:
                raise ValueError(
                    f"Confiance hors limites pour l'image {image_id}."
                )

            if (
                detection["x1"] < 0
                or detection["y1"] < 0
                or detection["x2"] < detection["x1"]
                or detection["y2"] < detection["y1"]
            ):
                raise ValueError(
                    f"Coordonnées bbox incorrectes pour l'image {image_id}."
                )

    print(f"Checkpoint validé : {len(data)} images.")

    return data


# ==================================================
# 4. VERIFICATION DE LA CONFIGURATION
# ==================================================

def verify_configuration():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [KB_IDS_FILE, IMAGE_DIR, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier ou dossier introuvable : {path}"
            )

    with open(KB_IDS_FILE, "r", encoding="utf-8") as file:
        image_ids = json.load(file)

    if not isinstance(image_ids, list):
        raise ValueError(
            "Le fichier des IDs doit contenir une liste."
        )

    if len(image_ids) != 10000:
        raise ValueError(
            f"10 000 images attendues, trouvé : {len(image_ids)}"
        )

    if any(
        isinstance(image_id, bool) or not isinstance(image_id, int)
        for image_id in image_ids
    ):
        raise ValueError(
            "Tous les IDs doivent être des entiers."
        )

    if any(image_id <= 0 for image_id in image_ids):
        raise ValueError("Les IDs doivent être positifs.")

    if len(set(image_ids)) != 10000:
        raise ValueError("Des IDs sont dupliqués.")

    missing_images = [
        image_id
        for image_id in image_ids
        if not (IMAGE_DIR / f"{image_id:012d}.jpg").is_file()
    ]

    if missing_images:
        raise ValueError(
            f"{len(missing_images)} images sont manquantes. "
            f"Exemples : {missing_images[:10]}"
        )

    print("Configuration vérifiée.")
    print(f"Images prévues : {len(image_ids)}")
    print(f"Seuil : {CONFIDENCE_THRESHOLD}")
    print(f"Appareil : {DEVICE}")
    print(f"Sauvegarde tous les : {SAVE_EVERY} images")
    print(f"Sortie : {OUTPUT_DIR}")

    return image_ids


# ==================================================
# 5. EXTRACTION YOLO
# ==================================================

def run_extraction(image_ids):

    detections = load_checkpoint()

    completed_ids = {
        item["image_id"] for item in detections
    }

    knowledge_base_ids = set(image_ids)

    if not completed_ids.issubset(knowledge_base_ids):
        raise ValueError(
            "Le checkpoint contient des IDs hors de la Knowledge Base."
        )

    remaining_ids = [
        image_id
        for image_id in image_ids
        if image_id not in completed_ids
    ]

    print(f"Images déjà traitées : {len(completed_ids)}")
    print(f"Images restantes : {len(remaining_ids)}")

    if remaining_ids:
        model = YOLO(str(MODEL_PATH))
    else:
        model = None
        print("Toutes les images sont déjà traitées.")

    start_time = time.time()

    for index, image_id in enumerate(remaining_ids, start=1):

        file_name = f"{image_id:012d}.jpg"
        image_path = IMAGE_DIR / file_name

        results = model.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        image_detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                image_detections.append({
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(confidence, 6),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

        detections.append({
            "image_id": image_id,
            "file_name": file_name,
            "detections": image_detections
        })

        if index % SAVE_EVERY == 0:

            save_json(detections, CHECKPOINT_FILE)

            print(
                f"Progression : "
                f"{len(completed_ids) + index}/"
                f"{len(image_ids)} images"
            )

    save_json(detections, CHECKPOINT_FILE)

    elapsed_time = time.time() - start_time

    save_csv(detections, OUTPUT_CSV)
    save_json(detections, OUTPUT_JSON)

    # Calcul des statistiques
    category_counts = Counter()
    total_detections = 0
    images_without_detections = 0
    confidence_values = []

    for image in detections:

        image_results = image["detections"]

        if not image_results:
            images_without_detections += 1

        for detection in image_results:

            category_counts[detection["class_name"]] += 1
            confidence_values.append(detection["confidence"])
            total_detections += 1

    statistics = {
        "total_images": len(detections),
        "total_detections": total_detections,

        "average_detections_per_image": (
            total_detections / len(detections)
            if detections else 0
        ),

        "images_without_detections": images_without_detections,
        "number_of_categories": len(category_counts),

        "mean_confidence": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else 0
        ),

        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "device": DEVICE,
        "elapsed_seconds": round(elapsed_time, 2),
        "category_counts": dict(category_counts)
    }

    save_json(statistics, STATISTICS_FILE)

    print("Extraction terminée.")
    print(f"Images traitées : {len(detections)}")
    print(f"Détections : {total_detections}")
    print(f"Durée : {elapsed_time:.2f} secondes")


# ==================================================
# 6. POINT D'ENTREE
# ==================================================

def main():

    image_ids = verify_configuration()

    if not RUN_EXTRACTION:
        print(
            "\nMODE VERIFICATION : "
            "aucune détection YOLO lancée."
        )
        print("La configuration est prête.")
        return

    run_extraction(image_ids)


if __name__ == "__main__":
    main()
