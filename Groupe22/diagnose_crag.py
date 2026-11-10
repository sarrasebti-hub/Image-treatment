import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

MODEL_NAME = "openai/clip-vit-base-patch32"

INDEX_FILE = EMBEDDINGS_DIR / "train2017.faiss"
TRAIN_IDS_FILE = EMBEDDINGS_DIR / "train2017_ids.json"
VAL_IDS_FILE = EMBEDDINGS_DIR / "val2017_ids.json"
VAL_EMB_FILE = EMBEDDINGS_DIR / "val2017_embeddings.npy"
KB_FILE = DATASET_DIR / "knowledge_base_train.json"

# Images de validation
IMAGE_DIR = DATASET_DIR / "val2017"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 20
QUERY_INDEX = 3
TOP_CAPTIONS = 10
MAX_TEXT_LENGTH = 77


# ============================================================
# 2. OUTILS
# ============================================================

def normalize_image_id(image_id):
    """Convertit un identifiant COCO en chaine de 12 chiffres."""
    return f"{int(image_id):012d}"


def check_required_files():

    required_files = [
        INDEX_FILE,
        TRAIN_IDS_FILE,
        VAL_IDS_FILE,
        VAL_EMB_FILE,
        KB_FILE,
    ]

    print("Verification des fichiers...")

    for file_path in required_files:

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

        print(f"OK : {file_path}")

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Dossier des images de validation introuvable : {IMAGE_DIR}"
        )

    print(f"OK : {IMAGE_DIR}")


# ============================================================
# 3. CHARGEMENT ET VERIFICATION DES DONNEES
# ============================================================

def load_data():

    print("\nChargement des fichiers...")

    index = faiss.read_index(str(INDEX_FILE))

    with open(TRAIN_IDS_FILE, "r", encoding="utf-8") as f:
        train_ids_raw = json.load(f)

    with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
        val_ids_raw = json.load(f)

    with open(KB_FILE, "r", encoding="utf-8") as f:
        knowledge_base_list = json.load(f)

    val_embeddings = np.load(
        VAL_EMB_FILE
    ).astype(np.float32)

    train_ids = [
        normalize_image_id(x)
        for x in train_ids_raw
    ]

    val_ids = [
        normalize_image_id(x)
        for x in val_ids_raw
    ]

    knowledge_base = {
        normalize_image_id(item["image_id"]): item
        for item in knowledge_base_list
    }

    if index.ntotal != len(train_ids):
        raise ValueError(
            "La taille de l'index FAISS ne correspond pas "
            "au nombre d'identifiants train."
        )

    if len(val_ids) != len(val_embeddings):
        raise ValueError(
            "Le nombre d'identifiants validation ne correspond pas "
            "au nombre d'embeddings validation."
        )

    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Des identifiants train sont dupliques.")

    if len(set(val_ids)) != len(val_ids):
        raise ValueError("Des identifiants validation sont dupliques.")

    if set(train_ids).intersection(val_ids):
        raise ValueError(
            "Fuite de donnees detectee : train et validation se chevauchent."
        )

    missing_ids = [
        image_id
        for image_id in train_ids
        if image_id not in knowledge_base
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} images train sont absentes de la base "
            f"de connaissances. Exemples : {missing_ids[:5]}"
        )

    if val_embeddings.ndim != 2 or val_embeddings.shape[1] != 512:
        raise ValueError(
            f"Dimensions inattendues des embeddings validation : "
            f"{val_embeddings.shape}"
        )

    print("Index FAISS :", index.ntotal)
    print("Images train :", len(train_ids))
    print("Images validation :", len(val_ids))
    print("Base de connaissances :", len(knowledge_base))
    print("Images train absentes de la base :", len(missing_ids))
    print("Separation train/validation : OK")

    return (
        index,
        train_ids,
        val_ids,
        val_embeddings,
        knowledge_base,
    )


# ============================================================
# 4. RECHERCHE FAISS
# ============================================================

def retrieve_images(
    index,
    query_embedding,
    train_ids,
    validation_ids,
    knowledge_base,
    top_k,
):

    query = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    if query.shape[1] != index.d:
        raise ValueError(
            f"Dimension de requete {query.shape[1]} incompatible "
            f"avec FAISS ({index.d})."
        )

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        raise ValueError("L'embedding de la requete est nul.")

    query = query / query_norm

    effective_k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query,
        effective_k
    )

    validation_set = set(validation_ids)

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        image_id = train_ids[int(idx)]

        if image_id in validation_set:
            continue

        entry = knowledge_base.get(image_id)

        if entry is None:
            continue

        captions = entry.get("captions", [])

        if not captions:
            continue

        retrieved.append({
            "image_id": image_id,
            "faiss_score": float(score),
            "captions": captions,
        })

    return retrieved


# ============================================================
# 5. ENCODAGE DES LEGENDES AVEC CLIP
# ============================================================

def encode_captions(captions, model, tokenizer, device):

    if not captions:
        return np.empty((0, 512), dtype=np.float32)

    embeddings = []

    with torch.inference_mode():

        for start in range(0, len(captions), 32):

            batch = captions[start:start + 32]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt",
            )

            tokens = {
                key: value.to(device)
                for key, value in tokens.items()
            }

            # Methode compatible avec notre version Transformers
            text_outputs = model.text_model(**tokens)

            text_features = model.text_projection(
                text_outputs.pooler_output
            )

            text_features = torch.nn.functional.normalize(
                text_features,
                p=2,
                dim=1,
            )

            embeddings.append(
                text_features.cpu().numpy()
            )

    return np.concatenate(
        embeddings,
        axis=0
    ).astype(np.float32)


# ============================================================
# 6. DIAGNOSTIC PRINCIPAL
# ============================================================

def main():

    print("=" * 65)
    print("DIAGNOSTIC CORRECTIVE RAG")
    print("=" * 65)

    check_required_files()

    (
        index,
        train_ids,
        val_ids,
        val_embeddings,
        knowledge_base,
    ) = load_data()

    if not 0 <= QUERY_INDEX < len(val_ids):
        raise IndexError(
            f"QUERY_INDEX={QUERY_INDEX} est invalide. "
            f"Choisir une valeur entre 0 et {len(val_ids) - 1}."
        )

    query_id = val_ids[QUERY_INDEX]

    query_embedding = val_embeddings[QUERY_INDEX]

    image_path = IMAGE_DIR / f"{query_id}.jpg"

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image de validation introuvable : {image_path}"
        )

    print("\nDevice :", DEVICE)
    print("Image de validation :", query_id)
    print("Chemin de l'image :", image_path)
    print("Nombre d'images demandees :", TOP_K)

    # --------------------------------------------------------
    # RECHERCHE FAISS
    # --------------------------------------------------------

    print("\nRecherche des images similaires avec FAISS...")

    retrieved = retrieve_images(
        index=index,
        query_embedding=query_embedding,
        train_ids=train_ids,
        validation_ids=val_ids,
        knowledge_base=knowledge_base,
        top_k=TOP_K,
    )

    if not retrieved:
        raise ValueError(
            "Aucune image valide recuperee par FAISS."
        )

    print("Images recuperees :", len(retrieved))

    # --------------------------------------------------------
    # CHARGEMENT DE CLIP
    # --------------------------------------------------------

    print("\nChargement de CLIP...")

    model = CLIPModel.from_pretrained(MODEL_NAME)

    processor = CLIPImageProcessor.from_pretrained(
        MODEL_NAME
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("CLIP charge avec succes sur :", DEVICE)

    # --------------------------------------------------------
    # ENCODAGE DE L'IMAGE
    # --------------------------------------------------------

    print("\nEncodage de l'image de validation...")

    with Image.open(image_path) as source_image:

        image = ImageOps.exif_transpose(
            source_image
        ).convert("RGB")

    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    image_inputs = {
        key: value.to(DEVICE)
        for key, value in image_inputs.items()
    }

    with torch.inference_mode():

        vision_outputs = model.vision_model(
            pixel_values=image_inputs["pixel_values"]
        )

        image_features = model.visual_projection(
            vision_outputs.pooler_output
        )

        image_features = torch.nn.functional.normalize(
            image_features,
            p=2,
            dim=1,
        )

    print("Embedding de l'image genere.")

    # --------------------------------------------------------
    # EVALUATION DES LEGENDES
    # --------------------------------------------------------

    print("\nEvaluation des legendes avec CLIP...")

    all_captions = []

    for result in retrieved:

        captions = result["captions"]

        text_features = encode_captions(
            captions,
            model,
            tokenizer,
            DEVICE,
        )

        similarities = (
            text_features @ image_features.cpu().numpy().T
        ).flatten()

        for caption, score in zip(captions, similarities):

            all_captions.append({
                "image_id": result["image_id"],
                "faiss_score": result["faiss_score"],
                "clip_score": float(score),
                "caption": caption,
            })

    if not all_captions:
        raise ValueError(
            "Aucune legende n'a pu etre evaluee."
        )

    # --------------------------------------------------------
    # CLASSEMENT DES LEGENDES
    # --------------------------------------------------------

    all_captions.sort(
        key=lambda item: item["clip_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # RESULTATS FAISS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("IMAGES RECUPEREES PAR FAISS")
    print("=" * 65)

    for rank, result in enumerate(retrieved, start=1):

        print(f"\n{rank}. Image ID : {result['image_id']}")

        print(
            f"   Score FAISS : {result['faiss_score']:.4f}"
        )

        print(
            f"   Nombre de legendes : {len(result['captions'])}"
        )

    # --------------------------------------------------------
    # TOP 10 DES LEGENDES
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(f"TOP {TOP_CAPTIONS} DES LEGENDES SELON CLIP")
    print("=" * 65)

    for rank, item in enumerate(
        all_captions[:TOP_CAPTIONS],
        start=1
    ):

        print(f"\n{rank}. Image source : {item['image_id']}")

        print(
            f"   Score FAISS : {item['faiss_score']:.4f}"
        )

        print(
            f"   Score CLIP : {item['clip_score']:.4f}"
        )

        print(
            f"   Legende : {item['caption']}"
        )

    # --------------------------------------------------------
    # STATISTIQUES
    # --------------------------------------------------------

    clip_scores = np.asarray(
        [item["clip_score"] for item in all_captions],
        dtype=np.float32,
    )

    print("\n" + "=" * 65)
    print("STATISTIQUES DU DIAGNOSTIC")
    print("=" * 65)

    print("Image de validation :", query_id)
    print("Images recuperees :", len(retrieved))
    print("Nombre total de legendes :", len(all_captions))

    print(
        f"Meilleur score CLIP : {float(np.max(clip_scores)):.4f}"
    )

    print(
        f"Score CLIP moyen : {float(np.mean(clip_scores)):.4f}"
    )

    print(
        f"Score CLIP minimum : {float(np.min(clip_scores)):.4f}"
    )

    print("\nDiagnostic termine.")


# ============================================================
# 7. POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    main()