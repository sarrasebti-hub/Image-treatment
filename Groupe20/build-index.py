import json
import time
from pathlib import Path

import faiss
import numpy as np

EMB = Path("embeddings")
OUT = Path("indexes")
OUT.mkdir(exist_ok=True)

img_emb = np.load(EMB / "kb_image_emb.npy").astype("float32")
txt_emb = np.load(EMB / "kb_text_emb.npy").astype("float32")
img_ids = json.load(open(EMB / "kb_image_ids.json"))
txt_img_ids = json.load(open(EMB / "kb_text_image_ids.json"))
dim = img_emb.shape[1]

# vectors must have length 1, so inner product = cosine similarity
for name, v in [("image", img_emb), ("text", txt_emb)]:
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"{name} vectors are not normalised"


def build(vectors, name):
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)            # flat = exact search, inner product
    index.add(np.ascontiguousarray(vectors))
    seconds = time.time() - t0
    path = OUT / f"kb_{name}.index"
    faiss.write_index(index, str(path))
    return index, {
        "index_type": "IndexFlatIP (flat, exact search)",
        "metric": "inner product (= cosine, vectors are L2-normalised)",
        "dimension": dim,
        "n_vectors": int(index.ntotal),
        "build_seconds": round(seconds, 4),
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "faiss_version": faiss.__version__,
    }


img_index, img_info = build(img_emb, "image")
txt_index, txt_info = build(txt_emb, "text")

# ---------- sanity checks ----------
# 1) image -> image: the nearest neighbour of a KB image is itself, with score ~1
D, I = img_index.search(img_emb[:200], 1)
assert (D > 0.99).all() and all(img_ids[I[k][0]] == img_ids[k] for k in range(200))

# 2) image -> caption: do the retrieved captions belong to the same image?
N = 1000
D, I = txt_index.search(img_emb[:N], 5)
hit1 = np.mean([txt_img_ids[I[k][0]] == img_ids[k] for k in range(N)])
hit5 = np.mean([img_ids[k] in {txt_img_ids[j] for j in I[k]} for k in range(N)])
print(f"image->caption Recall@1: {hit1:.3f} | Recall@5: {hit5:.3f} (first {N} KB images)")

info = {"image_index": img_info, "text_index": txt_info,
        "image_to_caption_recall_at_1": round(float(hit1), 3),
        "image_to_caption_recall_at_5": round(float(hit5), 3)}
json.dump(info, open(OUT / "index_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))