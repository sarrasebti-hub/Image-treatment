"""API de lecture de la release figée — c'est ce que les autres groupes importent.

    from rag_data import Release
    R = Release("artifacts")
    Q = R.image_embeddings("test")                    # (100, d) float32 normalisé, lignes alignées sur R.ids("test")
    scores, kb_img_ids = R.search_images(Q, k=5, query_ids=R.ids("test"))   # garde-fou anti-fuite intégré
    caps = R.kb_captions_of(kb_img_ids[0, 0])         # captions (brutes) de l'image retrouvée
    refs = R.eval_refs("test")                        # {image_id: [5 captions]} -> UNIQUEMENT pour l'évaluation

Le retriever n'a accès qu'à la KB : les références val/test sont dans des fichiers séparés.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Release:
    def __init__(self, root="artifacts", encoder=None):
        self.root = Path(root)
        if encoder is None:
            from common import load_config
            encoder = load_config()["primary_encoder"]
        self.encoder = encoder
        self.tables, self.emb, self.idx = self.root / "tables", self.root / "embeddings" / encoder, self.root / "index"
        self._kbc = None; self._meta = None; self._faiss = {}
        self.splits = {s: json.load(open(self.root / "splits" / f"{s}.json"))["images"] for s in ("kb", "val", "test")}

    # ---- identifiants & images
    def ids(self, split) -> np.ndarray:
        return np.load(self.emb / f"ids_{split}.npy")

    def image_path(self, split, image_id) -> Path:
        return self.root / "images_rgb" / split / f"{int(image_id):012d}.jpg"

    # ---- embeddings
    def image_embeddings(self, split) -> np.ndarray:
        return np.load(self.emb / f"{split}_img.npy")

    def kb_text_embeddings(self):
        """Embeddings texte (tour texte de l'encodeur) des captions KB + (caption_ids, image_ids)."""
        return (np.load(self.emb / "kb_txt.npy"), np.load(self.emb / "kb_txt_caption_ids.npy"),
                np.load(self.emb / "kb_txt_image_ids.npy"))

    def sbert_embeddings(self):
        return np.load(self.root / "embeddings" / "sbert_kb_txt.npy"), np.load(self.root / "embeddings" / "sbert_kb_txt_caption_ids.npy")

    # ---- captions
    def kb_captions(self) -> list[dict]:
        if self._kbc is None:
            self._kbc = _jsonl(self.tables / "kb_captions.jsonl")
            self._by_img = defaultdict(list)
            for c in self._kbc:
                self._by_img[c["image_id"]].append(c)
        return self._kbc

    def kb_captions_of(self, image_id, field="caption_raw") -> list[str]:
        self.kb_captions()
        return [c[field] for c in self._by_img[int(image_id)]]

    def eval_refs(self, split) -> dict[int, list[str]]:
        assert split in ("val", "test")
        out = defaultdict(list)
        for r in _jsonl(self.tables / f"eval_refs_{split}.jsonl"):
            out[r["image_id"]].append(r["caption_raw"])
        return dict(out)

    def metadata(self, split=None) -> dict[int, dict]:
        if self._meta is None:
            self._meta = {m["image_id"]: m for m in _jsonl(self.tables / "metadata.jsonl")}
        return {k: v for k, v in self._meta.items() if split is None or v["split"] == split}

    # ---- recherche
    def _index(self, kind):
        import faiss
        if kind not in self._faiss:
            self._faiss[kind] = faiss.read_index(str(self.idx / f"{self.encoder}_{kind}.faiss"))
        return self._faiss[kind]

    def search_images(self, Q: np.ndarray, k=5, query_ids=None, kind="flatip"):
        """Top-k images de la KB (cosinus). `query_ids` : IDs des images requêtes -> toute collision est refusée."""
        S, I = self._index(kind).search(np.ascontiguousarray(Q, dtype=np.float32), k)
        kb_ids = np.load(self.emb / "ids_kb.npy")
        out = kb_ids[I]
        if query_ids is not None:
            qi = np.asarray(query_ids)[:, None]
            assert not (out == qi).any(), "FUITE : une image requête a été retrouvée elle-même"
        return S, out

    def search_texts(self, Q: np.ndarray, k=5):
        """Image->texte : Q = embeddings image (même espace que la tour texte, CLIP/SigLIP). Renvoie (scores, caption_ids, image_ids)."""
        T, cids, iids = self.kb_text_embeddings()
        S, I = self._index("txt_flatip").search(np.ascontiguousarray(Q, dtype=np.float32), k)
        return S, cids[I], iids[I]

    def verify(self):
        """Compare les SHA-256 des fichiers à artifacts/manifest.json (garantit que tous les groupes ont la même release)."""
        from common import sha256_file
        man = json.load(open(self.root / "manifest.json")); bad = []
        for rel, info in man["files"].items():
            f = self.root / rel
            if not f.exists() or sha256_file(f) != info["sha256"]:
                bad.append(rel)
        return bad
