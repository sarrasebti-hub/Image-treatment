
import json
import random
import sys
from pathlib import Path

from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

KB_IDS_FILE = ROOT / "data/splits/knowledge_base_image_ids.json"
IMAGE_DIR = ROOT / "data/train2017"
MODEL_PATH = ROOT / "yolo26n.pt"

OUTPUT_DIR = ROOT / "data/visual_detections/checkpoint_test_5_fixed"

MANIFEST_FILE = OUTPUT_DIR / "selected_ids.json"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
FINAL_FILE = OUTPUT_DIR / "test_results.json"

CONFIDENCE = 0.25
DEVICE = "cpu"
TEST_SIZE = 5
FIRST_BATCH_SIZE = 3
RANDOM_SEED = 42


# ==========================================
# 2. SAUVEGARDE ET CHARGEMENT
# ==========================================

def save_json(data, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def load_json(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================
# 3. SELECTION DES 5 IMAGES
# ==========================================

def prepare_images():

    if not KB_IDS_FILE.exists():
        raise FileNotFoundError(KB_IDS_FILE)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_FILE.exists():
        image_ids = load_json(MANIFEST_FILE)
    else:
        kb_ids = load_json(KB_IDS_FILE)

        if not isinstance(kb_ids, list):
            raise ValueError("La Knowledge Base doit contenir une liste.")

        if len(kb_ids) < TEST_SIZE:
            raise ValueError("La Knowledge Base contient moins de 5 images.")

        random.seed(RANDOM_SEED)
        image_ids = random.sample(kb_ids, TEST_SIZE)

        save_json(image_ids, MANIFEST_FILE)

    if not isinstance(image_ids, list):
        raise ValueError("Manifeste invalide.")

    if len(image_ids) != TEST_SIZE:
        raise ValueError("Le manifeste doit contenir 5 images.")

    if len(set(image_ids)) != TEST_SIZE:
        raise ValueError("Le manifeste contient des IDs dupliqués.")

    for image_id in image_ids:
        if type(image_id) is not int:
            raise ValueError(f"ID invalide : {image_id}")

        image_path = IMAGE_DIR / f"{image_id:012d}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

    return image_ids


# ==========================================
# 4. DETECTION D'UNE IMAGE
# ==========================================

def detect_image(model, image_id):

    file_name = f"{image_id:012d}.jpg"
    image_path = IMAGE_DIR / file_name

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    objects = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            objects.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": round(confidence, 6),
                "bbox_xyxy": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    return {
        "image_id": image_id,
        "file_name": file_name,
        "number_of_objects": len(objects),
        "objects": objects
    }


# ==========================================
# 5. VALIDATION DU CHECKPOINT
# ==========================================

def validate_checkpoint(checkpoint, image_ids):

    if not isinstance(checkpoint, list):
        raise ValueError("Checkpoint invalide : une liste était attendue.")

    required_fields = {
        "image_id",
        "file_name",
        "number_of_objects",
        "objects"
    }

    completed_ids = set()

    for item in checkpoint:

        if not isinstance(item, dict):
            raise ValueError("Enregistrement invalide dans le checkpoint.")

        if not required_fields.issubset(item):
            raise ValueError("Champs manquants dans le checkpoint.")

        image_id = item["image_id"]

        if type(image_id) is not int:
            raise ValueError("ID invalide dans le checkpoint.")

        if image_id in completed_ids:
            raise ValueError(f"ID dupliqué : {image_id}")

        if image_id not in image_ids:
            raise ValueError(f"ID inconnu : {image_id}")

        if item["file_name"] != f"{image_id:012d}.jpg":
            raise ValueError(f"Nom de fichier incorrect : {image_id}")

        if not isinstance(item["objects"], list):
            raise ValueError(f"Liste d'objets invalide : {image_id}")

        if item["number_of_objects"] != len(item["objects"]):
            raise ValueError(f"Nombre d'objets incorrect : {image_id}")

        for obj in item["objects"]:

            if not isinstance(obj, dict):
                raise ValueError("Détection invalide.")

            if "bbox_xyxy" not in obj:
                raise ValueError("Coordonnées bbox manquantes.")

            bbox = obj["bbox_xyxy"]

            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Une bbox doit contenir 4 coordonnées.")

            if not all(
                isinstance(value, (int, float)) for value in bbox
            ):
                raise ValueError("Coordonnées bbox non numériques.")

        completed_ids.add(image_id)

    return completed_ids


# ==========================================
# 6. EXECUTION DES DEUX PHASES
# ==========================================

def main():

    if len(sys.argv) != 2 or sys.argv[1] not in ("phase1", "phase2"):
        print("Utilisation :")
        print("python scripts\\test_yolo_checkpoint_real.py phase1")
        print("python scripts\\test_yolo_checkpoint_real.py phase2")
        return

    phase = sys.argv[1]

    image_ids = prepare_images()

    checkpoint = load_json(CHECKPOINT_FILE)

    if checkpoint is None:
        checkpoint = []

    completed_ids = validate_checkpoint(checkpoint, image_ids)

    if phase == "phase1":
        target_ids = image_ids[:FIRST_BATCH_SIZE]
    else:
        if not CHECKPOINT_FILE.exists():
            raise FileNotFoundError(
                "Lance d'abord la phase1 pour créer le checkpoint."
            )

        target_ids = image_ids

    remaining_ids = [
        image_id for image_id in target_ids
        if image_id not in completed_ids
    ]

    print(f"Phase : {phase}")
    print(f"Images déjà sauvegardées : {len(completed_ids)}")
    print(f"Images à traiter : {len(remaining_ids)}")

    if remaining_ids:
        model = YOLO(str(MODEL_PATH))

        for image_id in remaining_ids:
            result = detect_image(model, image_id)
            checkpoint.append(result)

            save_json(checkpoint, CHECKPOINT_FILE)

            completed_ids.add(image_id)

            print(
                f"Image {image_id} sauvegardée | "
                f"Détections : {result['number_of_objects']}"
            )

    if phase == "phase1":
        print("\nPhase 1 terminée.")
        print("Checkpoint sauvegardé.")
        print("Tu peux maintenant lancer la phase2.")
        return

    final_ids = validate_checkpoint(checkpoint, image_ids)

    if len(checkpoint) != TEST_SIZE or final_ids != set(image_ids):
        raise ValueError(
            "Le test final ne contient pas les 5 images attendues."
        )

    save_json(checkpoint, FINAL_FILE)

    print("\n========== TEST DE REPRISE TERMINE ==========")
    print(f"Images sauvegardées : {len(checkpoint)}")
    print(
        f"Total des détections : "
        f"{sum(x['number_of_objects'] for x in checkpoint)}"
    )
    print(f"Résultats : {FINAL_FILE}")


if __name__ == "__main__":
    main()
