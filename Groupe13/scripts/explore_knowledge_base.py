
import json
import re
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CAPTIONS_FILE = ROOT / "data/captions/kb_captions_cleaned.json"
METADATA_FILE = ROOT / "data/metadata/kb_metadata.json"
SPLIT_FILE = ROOT / "data/splits/knowledge_base_image_ids.json"

IMAGE_EMBEDDINGS_FILE = ROOT / "data/embeddings/image_embeddings.npy"
IMAGE_IDS_FILE = ROOT / "data/embeddings/image_ids.npy"

OUTPUT_DIR = ROOT / "data/exploration"
REPORT_DIR = ROOT / "reports/exploration"

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPACY_MODEL = "en_core_web_sm"

BATCH_SIZE = 32
TOP_N = 20

EXPECTED_IMAGES = 10000
EXPECTED_EMBEDDING_DIM = 512

# False : vérification et préparation uniquement.
# True : autorise l'exécution complète de l'exploration.
RUN_EXPLORATION = True


# ============================================================
# 2. VOCABULAIRES
# ============================================================

COLORS = {
    "red": ["red", "crimson", "maroon"],
    "blue": ["blue", "navy"],
    "green": ["green"],
    "yellow": ["yellow", "gold"],
    "orange": ["orange"],
    "purple": ["purple", "violet"],
    "pink": ["pink"],
    "black": ["black"],
    "white": ["white"],
    "gray": ["gray", "grey"],
    "brown": ["brown"],
}

SHAPES = {
    "round": ["round", "circular", "circle"],
    "square": ["square"],
    "rectangular": ["rectangular", "rectangle"],
    "triangular": ["triangular", "triangle"],
    "oval": ["oval"],
    "long": ["long", "elongated"],
    "curved": ["curved"],
    "flat": ["flat"],
}

# Catégorie COCO -> supercatégorie officielle.
COCO_SUPERCATEGORIES = {
    "person": "person",

    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "airplane": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",

    "traffic light": "outdoor",
    "fire hydrant": "outdoor",
    "stop sign": "outdoor",
    "parking meter": "outdoor",
    "bench": "outdoor",

    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",

    "backpack": "accessory",
    "umbrella": "accessory",
    "handbag": "accessory",
    "tie": "accessory",
    "suitcase": "accessory",

    "frisbee": "sports",
    "skis": "sports",
    "snowboard": "sports",
    "sports ball": "sports",
    "kite": "sports",
    "baseball bat": "sports",
    "baseball glove": "sports",
    "skateboard": "sports",
    "surfboard": "sports",
    "tennis racket": "sports",

    "bottle": "kitchen",
    "wine glass": "kitchen",
    "cup": "kitchen",
    "fork": "kitchen",
    "knife": "kitchen",
    "spoon": "kitchen",
    "bowl": "kitchen",

    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",

    "chair": "furniture",
    "couch": "furniture",
    "potted plant": "furniture",
    "bed": "furniture",
    "dining table": "furniture",
    "toilet": "furniture",

    "tv": "electronic",
    "laptop": "electronic",
    "mouse": "electronic",
    "remote": "electronic",
    "keyboard": "electronic",
    "cell phone": "electronic",

    "microwave": "appliance",
    "oven": "appliance",
    "toaster": "appliance",
    "sink": "appliance",
    "refrigerator": "appliance",

    "book": "indoor",
    "clock": "indoor",
    "vase": "indoor",
    "scissors": "indoor",
    "teddy bear": "indoor",
    "hair drier": "indoor",
    "toothbrush": "indoor",
}


# ============================================================
# 3. OUTILS GENERAUX
# ============================================================

def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data, filename):
    path = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"JSON cree : {path}")


def save_csv(rows, filename):
    path = OUTPUT_DIR / filename

    if not rows:
        print(f"Aucune donnee pour : {filename}")
        return

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"CSV cree : {path}")


def save_bar_chart(labels, values, title, filename, top_n=None):
    if not labels:
        return

    if top_n is not None:
        labels = labels[:top_n]
        values = values[:top_n]

    plt.figure(figsize=(12, 7))
    plt.barh(labels[::-1], values[::-1])

    plt.title(title)
    plt.xlabel("Nombre")
    plt.tight_layout()

    path = REPORT_DIR / filename
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Graphique cree : {path}")


# ============================================================
# 4. VERIFICATION DES ENTREES
# ============================================================

def load_and_validate_data():

    print("\n========== VERIFICATION DES DONNEES ==========")

    required_files = [
        CAPTIONS_FILE,
        METADATA_FILE,
        SPLIT_FILE,
        IMAGE_EMBEDDINGS_FILE,
        IMAGE_IDS_FILE,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

        print(f"[OK] {path.name}")

    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        captions = json.load(f)

    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    with open(SPLIT_FILE, encoding="utf-8") as f:
        kb_ids_data = json.load(f)

    if not isinstance(captions, list):
        raise ValueError("Les captions doivent etre une liste.")

    if not isinstance(metadata, list):
        raise ValueError("Les metadonnees doivent etre une liste.")

    if isinstance(kb_ids_data, dict):
        kb_ids_data = kb_ids_data.get(
            "image_ids",
            kb_ids_data.get("ids", [])
        )

    if not isinstance(kb_ids_data, list):
        raise ValueError("Format du split Knowledge Base invalide.")

    kb_ids_list = list(map(int, kb_ids_data))
    kb_ids = set(kb_ids_list)

    if len(kb_ids) != len(kb_ids_list):
        raise ValueError("Le split contient des IDs dupliques.")

    metadata_by_id = {}

    for record in metadata:
        image_id = int(record["image_id"])

        if image_id in metadata_by_id:
            raise ValueError(
                f"ID metadata duplique : {image_id}"
            )

        metadata_by_id[image_id] = record

    caption_ids = set()

    for record in captions:
        if "image_id" not in record or "caption" not in record:
            raise ValueError("Structure de caption invalide.")

        caption_ids.add(int(record["image_id"]))

    metadata_ids = set(metadata_by_id)

    if len(kb_ids) != EXPECTED_IMAGES:
        raise ValueError(
            f"Nombre inattendu d'IDs KB : {len(kb_ids)}"
        )

    if metadata_ids != kb_ids:
        raise ValueError(
            "Les IDs des metadonnees ne correspondent pas "
            "au split Knowledge Base."
        )

    if not caption_ids.issubset(kb_ids):
        raise ValueError(
            "Certaines captions ne correspondent pas a la KB."
        )

    if not captions:
        raise ValueError("Aucune caption disponible.")

    image_embeddings = np.load(
        IMAGE_EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    image_ids = np.load(IMAGE_IDS_FILE)

    if image_embeddings.shape != (
        EXPECTED_IMAGES,
        EXPECTED_EMBEDDING_DIM
    ):
        raise ValueError(
            f"Dimensions image embeddings incorrectes : "
            f"{image_embeddings.shape}"
        )

    if len(image_ids) != EXPECTED_IMAGES:
        raise ValueError("Nombre incorrect d'IDs embeddings.")

    embedding_ids = list(map(int, image_ids))

    if len(set(embedding_ids)) != EXPECTED_IMAGES:
        raise ValueError("IDs embeddings dupliques.")

    if set(embedding_ids) != kb_ids:
        raise ValueError(
            "Les IDs embeddings ne correspondent pas au split KB."
        )

    if not np.isfinite(image_embeddings).all():
        raise ValueError(
            "Les embeddings contiennent des valeurs invalides."
        )

    print("\n[OK] Nombre d'images KB :", len(kb_ids))
    print("[OK] Nombre de captions :", len(captions))
    print("[OK] Nombre de metadonnees :", len(metadata))
    print("[OK] Dimensions embeddings :", image_embeddings.shape)
    print("[OK] Correspondance des IDs verifiee.")

    return (
        captions,
        metadata_by_id,
        image_embeddings,
        image_ids,
    )


# ============================================================
# 5. ANALYSE LINGUISTIQUE SPACY
# ============================================================

def analyze_pos(captions):

    print("\n========== ANALYSE LINGUISTIQUE ==========")

    nlp = spacy.load(SPACY_MODEL)

    pos_counter = Counter()
    nouns = Counter()
    verbs = Counter()
    adjectives = Counter()

    texts = [item["caption"] for item in captions]

    caption_count = 0

    for doc in nlp.pipe(texts, batch_size=256):

        caption_count += 1

        for token in doc:

            if token.is_space or token.is_punct:
                continue

            pos_counter[token.pos_] += 1

            lemma = token.lemma_.lower().strip()

            if not lemma or not lemma.isalpha():
                continue

            if token.pos_ == "NOUN":
                nouns[lemma] += 1

            elif token.pos_ == "VERB":
                verbs[lemma] += 1

            elif token.pos_ == "ADJ":
                adjectives[lemma] += 1

    result = {
        "number_of_captions": caption_count,
        "pos_distribution": dict(pos_counter.most_common()),
        "top_nouns": nouns.most_common(TOP_N),
        "top_verbs": verbs.most_common(TOP_N),
        "top_adjectives": adjectives.most_common(TOP_N),
    }

    save_json(result, "pos_analysis.json")

    save_csv(
        [
            {"pos": k, "count": v}
            for k, v in pos_counter.most_common()
        ],
        "pos_distribution.csv"
    )

    for name, counter in [
        ("nouns", nouns),
        ("verbs", verbs),
        ("adjectives", adjectives),
    ]:

        save_csv(
            [
                {"word": word, "count": count}
                for word, count in counter.most_common()
            ],
            f"top_{name}.csv"
        )

    save_bar_chart(
        list(pos_counter.keys()),
        list(pos_counter.values()),
        "POS distribution",
        "pos_distribution.png"
    )

    for name, counter in [
        ("nouns", nouns),
        ("verbs", verbs),
        ("adjectives", adjectives),
    ]:

        items = counter.most_common(TOP_N)

        save_bar_chart(
            [x[0] for x in items],
            [x[1] for x in items],
            f"Top {name}",
            f"top_{name}.png"
        )

    return result


# ============================================================
# 6. EXTRACTION DES COULEURS ET DES FORMES
# ============================================================

def count_lexical_features(captions, vocabulary):

    occurrence_counts = Counter()
    caption_counts = Counter()
    cooccurrences = Counter()

    compiled_patterns = {
        label: [
            re.compile(
                r"(?<![a-z])"
                + re.escape(variant)
                + r"(?![a-z])",
                re.IGNORECASE
            )
            for variant in variants
        ]
        for label, variants in vocabulary.items()
    }

    for item in captions:

        text = item["caption"].lower()
        found = set()

        for label, patterns in compiled_patterns.items():

            occurrences = sum(
                len(pattern.findall(text))
                for pattern in patterns
            )

            if occurrences > 0:
                occurrence_counts[label] += occurrences
                caption_counts[label] += 1
                found.add(label)

        for first, second in combinations(sorted(found), 2):
            cooccurrences[(first, second)] += 1

    return occurrence_counts, caption_counts, cooccurrences


def analyze_colors(captions):

    print("\n========== ANALYSE DES COULEURS ==========")

    counts, caption_counts, cooccurrences = (
        count_lexical_features(captions, COLORS)
    )

    result = {
        "method": "Lexical extraction from captions",
        "warning": (
            "Results represent color mentions in captions, "
            "not colors detected from image pixels."
        ),
        "color_mentions": dict(counts.most_common()),
        "captions_per_color": dict(caption_counts.most_common()),
        "color_cooccurrences": [
            {
                "color_1": pair[0],
                "color_2": pair[1],
                "count": count,
            }
            for pair, count in cooccurrences.most_common()
        ],
    }

    save_json(result, "color_analysis.json")

    save_csv(
        [
            {
                "color": color,
                "mentions": counts[color],
                "captions": caption_counts[color],
            }
            for color in COLORS
        ],
        "color_analysis.csv"
    )

    save_bar_chart(
        list(counts.keys()),
        list(counts.values()),
        "Colors mentioned in captions",
        "color_distribution.png"
    )

    return result


def analyze_shapes(captions):

    print("\n========== ANALYSE DES FORMES ==========")

    counts, caption_counts, cooccurrences = (
        count_lexical_features(captions, SHAPES)
    )

    result = {
        "method": "Lexical extraction from captions",
        "warning": (
            "Results represent shape descriptions in captions, "
            "not geometric measurements from images."
        ),
        "shape_mentions": dict(counts.most_common()),
        "captions_per_shape": dict(caption_counts.most_common()),
        "shape_cooccurrences": [
            {
                "shape_1": pair[0],
                "shape_2": pair[1],
                "count": count,
            }
            for pair, count in cooccurrences.most_common()
        ],
    }

    save_json(result, "shape_analysis.json")

    save_csv(
        [
            {
                "shape": shape,
                "mentions": counts[shape],
                "captions": caption_counts[shape],
            }
            for shape in SHAPES
        ],
        "shape_analysis.csv"
    )

    save_bar_chart(
        list(counts.keys()),
        list(counts.values()),
        "Shapes mentioned in captions",
        "shape_distribution.png"
    )

    return result


# ============================================================
# 7. ENCODAGE TEXTUEL CLIP
# ============================================================

def encode_clip_texts(model, processor, texts):

    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        features = model.text_projection(pooled_output)

        features = features / features.norm(
            p=2,
            dim=-1,
            keepdim=True
        )

    return features.cpu().numpy()


# ============================================================
# 8. CLASSIFICATION CLIP
# ============================================================

def analyze_clip_binary(
    model,
    processor,
    image_embeddings,
    image_ids,
    metadata_by_id,
    prompts,
    labels,
    output_name,
    chart_name,
):

    print(f"\n========== ANALYSE {output_name.upper()} ==========")

    text_embeddings = encode_clip_texts(
        model,
        processor,
        prompts
    )

    scores = np.asarray(image_embeddings) @ text_embeddings.T

    predictions = np.argmax(scores, axis=1)

    records = []

    for index, image_id in enumerate(image_ids):

        image_id = int(image_id)

        first_score = float(scores[index, 0])
        second_score = float(scores[index, 1])

        predicted_label = labels[int(predictions[index])]

        record = {
            "image_id": image_id,
            "file_name": metadata_by_id[image_id]["file_name"],
            f"{labels[0].lower()}_similarity": first_score,
            f"{labels[1].lower()}_similarity": second_score,
            "predicted_class": predicted_label,
            "similarity_margin": abs(first_score - second_score),
        }

        records.append(record)

    distribution = Counter(
        record["predicted_class"]
        for record in records
    )

    result = {
        "method": "CLIP cosine similarity",
        "model": MODEL_NAME,
        "number_of_images": len(records),
        "prompts": prompts,
        "class_distribution": dict(distribution),
        "mean_similarity": {
            labels[0]: float(np.mean(scores[:, 0])),
            labels[1]: float(np.mean(scores[:, 1])),
        },
        "mean_similarity_margin": float(
            np.mean(np.abs(scores[:, 0] - scores[:, 1]))
        ),
        "warning": (
            "Similarity scores are not probabilities or verified "
            "ground-truth labels. Predictions use the higher "
            "cosine similarity."
        ),
        "predictions": records,
    }

    save_json(result, output_name)

    save_csv(
        records,
        output_name.replace(".json", ".csv")
    )

    save_bar_chart(
        list(distribution.keys()),
        list(distribution.values()),
        output_name.replace(".json", " distribution"),
        chart_name
    )

    return result


# ============================================================
# 9. ANALYSE DES OBJETS COCO
# ============================================================

def analyze_coco_categories(metadata_by_id):

    print("\n========== ANALYSE DES CATEGORIES COCO ==========")

    category_counts = Counter()
    supercategory_counts = Counter()
    image_object_counts = []

    category_cooccurrences = Counter()
    category_id_to_name = {}

    total_objects = 0
    images_without_objects = 0

    for image_id, record in metadata_by_id.items():

        objects = record.get("objects", [])

        total_objects += len(objects)

        image_object_counts.append({
            "image_id": image_id,
            "object_count": len(objects),
        })

        if not objects:
            images_without_objects += 1

        image_categories = set()

        for obj in objects:

            category = obj["category"]
            category_id = int(obj["category_id"])

            previous_name = category_id_to_name.get(category_id)

            if previous_name is not None and previous_name != category:
                raise ValueError(
                    f"ID {category_id} associe a plusieurs noms."
                )

            category_id_to_name[category_id] = category

            category_counts[category] += 1

            supercategory = COCO_SUPERCATEGORIES.get(category)

            if supercategory is None:
                raise ValueError(
                    f"Supercategorie inconnue : {category}"
                )

            supercategory_counts[supercategory] += 1

            image_categories.add(category_id)

        for first, second in combinations(sorted(image_categories), 2):
            category_cooccurrences[(first, second)] += 1

    average_objects = (
        total_objects / len(metadata_by_id)
        if metadata_by_id else 0
    )

    result = {
        "number_of_images": len(metadata_by_id),
        "total_objects": total_objects,
        "average_objects_per_image": average_objects,
        "images_without_objects": images_without_objects,
        "category_counts": dict(category_counts.most_common()),
        "supercategory_counts": dict(supercategory_counts.most_common()),
        "category_cooccurrences": [
            {
                "category_id_1": pair[0],
                "category_name_1": category_id_to_name[pair[0]],
                "category_id_2": pair[1],
                "category_name_2": category_id_to_name[pair[1]],
                "images": count,
            }
            for pair, count in category_cooccurrences.most_common()
        ],
    }

    save_json(result, "coco_categories_analysis.json")

    save_csv(
        [
            {"category": k, "count": v}
            for k, v in category_counts.most_common()
        ],
        "coco_categories.csv"
    )

    save_csv(
        [
            {"supercategory": k, "count": v}
            for k, v in supercategory_counts.most_common()
        ],
        "coco_supercategories.csv"
    )

    save_csv(
        image_object_counts,
        "coco_objects_per_image.csv"
    )

    save_csv(
        [
            {
                "category_id_1": pair[0],
                "category_name_1": category_id_to_name[pair[0]],
                "category_id_2": pair[1],
                "category_name_2": category_id_to_name[pair[1]],
                "images": count,
            }
            for pair, count in category_cooccurrences.most_common()
        ],
        "coco_category_cooccurrences.csv"
    )

    save_bar_chart(
        list(category_counts.keys()),
        list(category_counts.values()),
        "COCO object categories",
        "coco_categories.png",
        top_n=TOP_N
    )

    save_bar_chart(
        list(supercategory_counts.keys()),
        list(supercategory_counts.values()),
        "COCO supercategories",
        "coco_supercategories.png"
    )

    return result


# ============================================================
# 10. RESUME GLOBAL
# ============================================================

def create_global_summary(
    captions,
    metadata_by_id,
    pos_result,
    color_result,
    shape_result,
    indoor_result,
    day_night_result,
    coco_result,
    elapsed_time,
):

    summary = {
        "project": "COCO Knowledge Base Exploration",
        "scope": "Knowledge Base only",
        "number_of_images": len(metadata_by_id),
        "number_of_captions": len(captions),
        "analyses": {
            "linguistic": "spaCy POS tagging",
            "colors": "Lexical extraction from captions",
            "shapes": "Lexical extraction from captions",
            "indoor_outdoor": "CLIP cosine similarity",
            "day_night": "CLIP cosine similarity",
            "coco_objects": "COCO annotations",
        },
        "indoor_outdoor_distribution": (
            indoor_result["class_distribution"]
        ),
        "day_night_distribution": (
            day_night_result["class_distribution"]
        ),
        "total_coco_objects": coco_result["total_objects"],
        "average_objects_per_image": (
            coco_result["average_objects_per_image"]
        ),
        "elapsed_seconds": elapsed_time,
        "limitations": [
            "Colors and shapes are extracted from text, "
            "not detected from image pixels.",
            "CLIP similarities are not probabilities.",
            "Indoor/outdoor and day/night labels are predictions, "
            "not verified ground truth.",
            "COCO categories represent annotated objects, "
            "not every object present in an image.",
        ],
    }

    save_json(summary, "exploration_summary.json")

    print("\n========== RESUME FINAL ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ============================================================
# 11. EXECUTION PRINCIPALE
# ============================================================

def main():

    start_time = time.time()

    print("\n==============================================")
    print("   COCO KNOWLEDGE BASE EXPLORATION")
    print("==============================================")

    ensure_directories()

    (
        captions,
        metadata_by_id,
        image_embeddings,
        image_ids,
    ) = load_and_validate_data()

    if not RUN_EXPLORATION:
        print("\nMODE PREPARATION UNIQUEMENT")
        print("Les donnees sont verifiees.")
        print("Aucune analyse n'a ete lancee.")
        print("RUN_EXPLORATION = False")
        return

    print("\nChargement du modele spaCy...")
    spacy.load(SPACY_MODEL)

    pos_result = analyze_pos(captions)
    color_result = analyze_colors(captions)
    shape_result = analyze_shapes(captions)

    print("\nChargement du modele CLIP...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    print("Modele CLIP charge sur :", DEVICE)

    indoor_result = analyze_clip_binary(
        model=model,
        processor=processor,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        metadata_by_id=metadata_by_id,
        prompts=[
            "A photo of an indoor scene.",
            "A photo of an outdoor scene.",
        ],
        labels=["Indoor", "Outdoor"],
        output_name="indoor_outdoor_predictions.json",
        chart_name="indoor_outdoor_distribution.png",
    )

    day_night_result = analyze_clip_binary(
        model=model,
        processor=processor,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        metadata_by_id=metadata_by_id,
        prompts=[
            "A photo taken during daytime.",
            "A photo taken at night.",
        ],
        labels=["Day", "Night"],
        output_name="day_night_predictions.json",
        chart_name="day_night_distribution.png",
    )

    coco_result = analyze_coco_categories(metadata_by_id)

    elapsed_time = time.time() - start_time

    create_global_summary(
        captions,
        metadata_by_id,
        pos_result,
        color_result,
        shape_result,
        indoor_result,
        day_night_result,
        coco_result,
        elapsed_time,
    )

    print("\n========== EXPLORATION TERMINEE ==========")
    print(f"Duree : {elapsed_time:.2f} secondes")


if __name__ == "__main__":
    main()
