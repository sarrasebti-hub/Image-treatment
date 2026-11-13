
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = DATA / "splits"
ANNOTATIONS = DATA / "annotations"

def load_ids(filename):
    with open(SPLITS / filename, "r", encoding="utf-8") as f:
        return set(json.load(f))

def load_coco_ids(filename):
    print(f"Lecture de {filename}...")
    with open(ANNOTATIONS / filename, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return {img["id"] for img in coco["images"]}

# Anciens splits
kb = load_ids("knowledge_base_image_ids.json")
validation = load_ids("validation_image_ids.json")
test = load_ids("test_image_ids.json")

# Identifiants COCO officiels
train_ids = load_coco_ids("instances_train2017.json")
val_ids = load_coco_ids("instances_val2017.json")

print("\n========== TAILLES ==========")
print("Knowledge Base :", len(kb))
print("Validation :", len(validation))
print("Ancien test :", len(test))

print("\n========== PROVENANCE ==========")
print("KB dans train2017 :", len(kb & train_ids), "/", len(kb))
print("Validation dans train2017 :", len(validation & train_ids), "/", len(validation))
print("Test dans val2017 :", len(test & val_ids), "/", len(test))

print("\n========== CHEVAUCHEMENTS ==========")
print("KB / Validation :", len(kb & validation))
print("KB / Test :", len(kb & test))
print("Validation / Test :", len(validation & test))

print("\n========== VALIDATION ==========")
print("KB correcte :", len(kb) == 10000 and kb <= train_ids)
print("Validation correcte :", len(validation) == 100 and validation <= train_ids)
print("Test correct :", len(test) == 100 and test <= val_ids)
print("KB et validation disjointes :", not bool(kb & validation))

print("\nVérification terminée. Aucun fichier modifié.")
