import json
from pathlib import Path
import torch
import numpy as np
from transformers import CLIPModel, CLIPTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "dataset" / "embeddings"

OLD_FILE = EMBEDDINGS_DIR / "evaluation_results.json"
NEW_FILE = EMBEDDINGS_DIR / "crag_diversification_results.json"

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(OLD_FILE, "r", encoding="utf-8") as f:
    old_data = json.load(f)

with open(NEW_FILE, "r", encoding="utf-8") as f:
    new_data = json.load(f)

old_map = {x["query_id"]: x for x in old_data}
new_map = {x["query_image_id"]: x for x in new_data}

common_ids = sorted(set(old_map) & set(new_map))

print("Common queries:", len(common_ids))

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
model.eval()


def encode_captions(captions):

    tokens = tokenizer(
        captions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.text_model(**tokens)
        features = model.text_projection(outputs.pooler_output)

        features = features / features.norm(
            dim=-1, keepdim=True
        )

    return features.cpu().numpy()


def calculate_ild(captions):

    if len(captions) < 2:
        return 0.0

    embeddings = encode_captions(captions)

    similarity_matrix = embeddings @ embeddings.T

    n = len(captions)

    distances = 1 - similarity_matrix

    mask = ~np.eye(n, dtype=bool)

    return float(distances[mask].mean())


methods = {
    "CLIP Reranking": [],
    "MMR Lambda 0.7": [],
    "MMR Lambda 0.9": []
}

for query_id in common_ids:

    old = old_map[query_id]
    new = new_map[query_id]

    reranking_captions = [
        x["caption"] for x in old["reranking"]
    ]

    mmr_07_captions = [
        x["caption"] for x in old["mmr"]
    ]

    mmr_09_captions = [
        x["caption"] for x in new["selected_captions"]
    ]

    methods["CLIP Reranking"].append(
        calculate_ild(reranking_captions)
    )

    methods["MMR Lambda 0.7"].append(
        calculate_ild(mmr_07_captions)
    )

    methods["MMR Lambda 0.9"].append(
        calculate_ild(mmr_09_captions)
    )

print("\n========== INTRA-LIST DIVERSITY ==========")

for method, values in methods.items():

    print(f"\n{method}")
    print("Mean ILD:", round(float(np.mean(values)), 4))

print("\n==========================================")