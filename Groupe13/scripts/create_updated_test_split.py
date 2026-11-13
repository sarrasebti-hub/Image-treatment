
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = DATA / "splits"
ANNOTATIONS = DATA / "annotations"

SEED = 42
TEST_SIZE = 1000

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Charger les anciens identifiants
kb = set(read_json(SPLITS / "knowledge_base_image_ids.json"))
validation = set(read_json(SPLITS / "validation_image_ids.json"))
old_test_list = read_json(SPLITS / "test_image_ids.json")
old_test = set(old_test_list)

# Charger les images officielles de COCO val2017
coco_val = read_json(ANNOTATIONS / "instances_val2017.json")
val_ids = {image["id"] for image in coco_val["images"]}

# Vérifications avant toute création
assert len(kb) == 10000, "La KB ne contient pas 10000 images."
assert len(validation) == 100, "La validation ne contient pas 100 images."
assert len(old_test) == 100, "L'ancien test ne contient pas 100 images."
assert old_test <= val_ids, "Certaines anciennes images ne sont pas dans val2017."
assert not (kb & validation), "Chevauchement KB/validation."
assert len(val_ids - old_test) >= TEST_SIZE - len(old_test)

# Conserver l'ancien test sous un nouveau nom
legacy_path = SPLITS / "test_100_image_ids_legacy.json"

if legacy_path.exists():
    print("Copie legacy déjà existante : elle sera conservée.")
else:
    save_json(legacy_path, old_test_list)
    print("Ancien test sauvegardé :", legacy_path.name)

# Ajouter 900 nouvelles images aux 100 anciennes
remaining_ids = sorted(val_ids - old_test)

rng = random.Random(SEED)
additional_ids = rng.sample(remaining_ids, TEST_SIZE - len(old_test))

new_test = sorted(old_test | set(additional_ids))

# Enregistrer dans un nouveau fichier
new_path = SPLITS / "test_1000_image_ids.json"
save_json(new_path, new_test)

# Vérification finale
assert len(new_test) == 1000
assert old_test <= set(new_test)
assert set(new_test) <= val_ids

print("\n========== NOUVEAU TEST ==========")
print("Ancien test :", len(old_test))
print("Images ajoutées :", len(additional_ids))
print("Nouveau test :", len(new_test))
print("Images issues de val2017 :", len(set(new_test) & val_ids))
print("Ancien test inclus :", old_test <= set(new_test))
print("Fichier créé :", new_path)
print("\nAncien fichier test_image_ids.json non modifié.")
