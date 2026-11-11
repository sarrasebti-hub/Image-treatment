"""
Member 3 - Image Caption Generation Pipeline
Group 11: RAG Reranking

This module provides a reusable CaptionGenerator class for generating
baseline and RAG-enhanced captions using BLIP-2 + Member 2's FAISS index.

Usage:
    from src.captioning.generate_pipeline import CaptionGenerator
    gen = CaptionGenerator(repo_dir="path/to/repo")
    caption = gen.generate_rag("path/to/image.jpg")
"""

import json
import torch
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)
from sentence_transformers import CrossEncoder


class CaptionGenerator:
    """End-to-end caption generator with Baseline and RAG modes."""

    def __init__(self, repo_dir):
        self.repo_dir = Path(repo_dir)

        print("Loading BLIP-2...")
        self.blip_processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-flan-t5-xl"
        )
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.blip_model.eval()

        print("Loading CLIP (for retrieval)...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(
            str(self.repo_dir / "outputs" / "faiss_index.bin")
        )
        with open(self.repo_dir / "outputs" / "valid_kb_ids.json", "r") as f:
            self.valid_kb_ids = json.load(f)
        with open(self.repo_dir / "outputs" / "captions.json", "r") as f:
            captions_data = json.load(f)

        self.caption_lookup = {}
        for item in captions_data:
            img_id = item["image_id"]
            self.caption_lookup.setdefault(img_id, []).append(
                item["caption_original"]
            )

        print("Loading Reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("✅ CaptionGenerator ready!")

    def get_rag_examples(self, image_path, top_k=5):
        """Retrieve top-k similar captions using FAISS + Cross-Encoder."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs).pooler_output
            vec = features.numpy().flatten()
            vec = vec / np.linalg.norm(vec)
            vec = np.array([vec]).astype("float32")

        # Stage 1: FAISS - top 20
        distances, indices = self.faiss_index.search(vec, 20)

        candidates = []
        for idx in indices[0]:
            kb_id = self.valid_kb_ids[idx]
            if kb_id in self.caption_lookup:
                candidates.append(self.caption_lookup[kb_id][0])

        # Stage 2: Cross-Encoder reranking - top 5
        pairs = [["Describe this image", c] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), reverse=True)
        return [text for _, text in ranked[:top_k]]

    def generate_baseline(self, image_path):
        """Generate caption without RAG."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt").to(
            self.blip_model.device
        )
        with torch.no_grad():
            ids = self.blip_model.generate(**inputs, max_new_tokens=50)
        return self.blip_processor.batch_decode(ids, skip_special_tokens=True)[0]

    def generate_rag(self, image_path):
        """Generate caption using top-5 retrieved examples."""
        examples = self.get_rag_examples(image_path, top_k=5)
        examples_text = "\n".join(f"- {ex}" for ex in examples)
        prompt = (
            f"Here are some examples of similar images:\n{examples_text}\n\n"
            "Now, describe the new image based on the visual details, "
            "using the style of the examples above."
        )
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.blip_model.device)
        with torch.no_grad():
            ids = self.blip_model.generate(**inputs, max_new_tokens=50)
        return self.blip_processor.batch_decode(ids, skip_special_tokens=True)[0]


# --- Quick test ---
if __name__ == "__main__":
    repo = Path("..")  # Adjust if needed
    gen = CaptionGenerator(repo)
    test_image = "test.jpg"
    print("Baseline:", gen.generate_baseline(test_image))
    print("RAG:", gen.generate_rag(test_image))