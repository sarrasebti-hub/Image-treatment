
import json
import random
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Fichiers d'entrée
SPLIT_FILE = ROOT / "data" / "splits" / "knowledge_base_image_ids.json"
IMAGE_DIR = ROOT / "data" / "train2017"

# Nouveau dossier de sortie : les anciens résultats restent intacts
OUTPUT_DIR = ROOT / "data" / "visual_detections_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Charger les identifiants de la Knowledge Base
with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    image_ids = json.load(f)

# Sélection déterministe de 5 images
random.seed(42)
selected_ids = random.sample(image_ids, 5)

print("Images sélectionnées :", selected_ids)

# Charger le modèle YOLO26n
model = YOLO("yolo26n.pt")

all_results = []

for image_id in selected_ids:
    image_path = IMAGE_DIR / f"{image_id:012d}.jpg"

    if not image_path.exists():
        print(f"Image introuvable : {image_path}")
        continue

    print(f"\nDétection sur : {image_path.name}")

    results = model.predict(
        source=str(image_path),
        device="cpu",
        conf=0.25,
        verbose=False
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": round(confidence, 4),
            "bbox_xyxy": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ]
        })

    all_results.append({
        "image_id": image_id,
        "image_file": image_path.name,
        "number_of_objects": len(detections),
        "objects": detections
    })

    # Sauvegarder une image annotée
    annotated = result.plot()
    from PIL import Image
    Image.fromarray(annotated[:, :, ::-1]).save(
        OUTPUT_DIR / f"{image_id:012d}_detected.jpg"
    )

    print(f"Objets détectés : {len(detections)}")

# Sauvegarder les détections au format JSON
json_path = OUTPUT_DIR / "yolo_test_results.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print("\n========== TEST TERMINÉ ==========")
print("Images traitées :", len(all_results))
print("Total objets détectés :", sum(
    item["number_of_objects"] for item in all_results
))
print("Résultats :", json_path)
