
from pathlib import Path
import json
import random

# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_CAPTIONS = ANNOTATIONS_DIR / "captions_train2017.json"
VAL_CAPTIONS = ANNOTATIONS_DIR / "captions_val2017.json"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

N_KB = 10_000
N_VALIDATION = 100

# À confirmer avec l'enseignante : 100 ou 1000
N_TEST = 100


# ==================================================
# 2. CHARGER LES IDENTIFIANTS
# ==================================================

def load_image_ids(json_path):

    if not json_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {json_path}"
        )

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    image_ids = [
        image["id"] for image in data["images"]
    ]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(
            f"Des IDs d'images sont dupliqués dans {json_path.name}"
        )

    return image_ids


# ==================================================
# 3. CRÉER LES SPLITS
# ==================================================

def create_splits():

    if N_TEST is None:
        raise ValueError(
            "N_TEST n'est pas défini. "
            "Confirme d'abord le nombre d'images de test."
        )

    random.seed(SEED)

    train_ids = load_image_ids(TRAIN_CAPTIONS)
    test_source_ids = load_image_ids(VAL_CAPTIONS)

    if len(train_ids) < N_KB + N_VALIDATION:
        raise ValueError(
            "Pas assez d'images dans train2017."
        )

    if len(test_source_ids) < N_TEST:
        raise ValueError(
            "Pas assez d'images dans val2017."
        )

    # Mélanger les IDs du train
    shuffled_train = train_ids.copy()
    random.shuffle(shuffled_train)

    # Knowledge Base
    kb_ids = shuffled_train[:N_KB]

    # Validation, sans chevauchement avec la KB
    validation_ids = shuffled_train[
        N_KB:N_KB + N_VALIDATION
    ]

    # Test provenant de val2017
    shuffled_test = test_source_ids.copy()
    random.shuffle(shuffled_test)

    test_ids = shuffled_test[:N_TEST]

    # Vérification anti-leakage
    assert set(kb_ids).isdisjoint(validation_ids)
    assert set(kb_ids).isdisjoint(test_ids)
    assert set(validation_ids).isdisjoint(test_ids)

    # Sauvegarde
    splits = {
        "knowledge_base": kb_ids,
        "validation": validation_ids,
        "test": test_ids
    }

    for name, ids in splits.items():

        output_file = SPLITS_DIR / f"{name}_image_ids.json"

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(ids, file, indent=4)

        print(f"{name}: {len(ids)} images")
        print(f"Saved: {output_file}")

    print("\nAnti-leakage verification: PASSED")
    print(f"Random seed: {SEED}")


# ==================================================
# 4. MAIN
# ==================================================

if __name__ == "__main__":
    create_splits()
