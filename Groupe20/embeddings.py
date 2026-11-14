import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-base-patch32"
BATCH = 64
TEXT_BATCH = 256
OUT = Path("embeddings")
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = CLIPModel.from_pretrained(MODEL).to(device).eval()
proc = CLIPProcessor.from_pretrained(MODEL)

kb = json.load(open("processed/kb_captions.json"))
ids = [row["image_id"] for row in kb]


def load_image(image_id):
    img = Image.open(f"images/kb/{image_id:012d}.jpg")
    img = ImageOps.exif_transpose(img)      # fix rotation
    return img.convert("RGB")                # force RGB


def normalise(x):
    return x / x.norm(dim=-1, keepdim=True)


def embed_images(pixel_values):
    """Vision tower -> projection -> 512-d vector (works on any transformers version)."""
    pooled = model.vision_model(pixel_values=pixel_values).pooler_output
    return model.visual_projection(pooled)


def embed_texts(input_ids, attention_mask):
    """Text tower -> projection -> 512-d vector."""
    pooled = model.text_model(input_ids=input_ids,
                              attention_mask=attention_mask).pooler_output
    return model.text_projection(pooled)


# ---------- images ----------
t0 = time.time()
img_emb = []
with torch.no_grad():
    for s in range(0, len(ids), BATCH):
        batch = [load_image(i) for i in ids[s:s + BATCH]]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        img_emb.append(normalise(embed_images(inputs["pixel_values"])).cpu())
        if (s // BATCH) % 20 == 0:
            print(f"images: {s}/{len(ids)}")
img_emb = torch.cat(img_emb).numpy().astype("float32")
t_img = time.time() - t0
assert img_emb.shape == (len(ids), 512), f"unexpected image embedding shape: {img_emb.shape}"

# ---------- captions (5 per image, cleaned version) ----------
texts = [c for row in kb for c in row["caption_clean"]]
text_image_ids = [row["image_id"] for row in kb for _ in row["caption_clean"]]

t0 = time.time()
txt_emb = []
with torch.no_grad():
    for s in range(0, len(texts), TEXT_BATCH):
        inputs = proc(text=texts[s:s + TEXT_BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=77).to(device)
        txt_emb.append(
            normalise(embed_texts(inputs["input_ids"], inputs["attention_mask"])).cpu()
        )
        if (s // TEXT_BATCH) % 20 == 0:
            print(f"captions: {s}/{len(texts)}")
txt_emb = torch.cat(txt_emb).numpy().astype("float32")
t_txt = time.time() - t0
assert txt_emb.shape == (len(texts), 512), f"unexpected text embedding shape: {txt_emb.shape}"

# ---------- save ----------
np.save(OUT / "kb_image_emb.npy", img_emb)
np.save(OUT / "kb_text_emb.npy", txt_emb)
json.dump(ids, open(OUT / "kb_image_ids.json", "w"))
json.dump(text_image_ids, open(OUT / "kb_text_image_ids.json", "w"))

info = {
    "model": MODEL,
    "device": device,
    "embedding_dim": int(img_emb.shape[1]),
    "normalised": True,
    "batch_size_images": BATCH,
    "batch_size_text": TEXT_BATCH,
    "n_images": int(img_emb.shape[0]),
    "n_captions": int(txt_emb.shape[0]),
    "seconds_images": round(t_img, 1),
    "seconds_text": round(t_txt, 1),
    "torch": torch.__version__,
}
json.dump(info, open(OUT / "embedding_info.json", "w"), indent=1)
print(info)

# ---------- quality control ----------
assert not np.isnan(img_emb).any() and not np.isnan(txt_emb).any(), "NaN found!"
assert len(set(ids)) == len(ids), "duplicate image IDs!"
print("Quality control passed.")