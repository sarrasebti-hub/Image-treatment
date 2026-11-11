import json
import numpy as np
import torch
import faiss
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RAGImageCaptioner:
    def __init__(self, model_id="google/flan-t5-small", embedding_dim=512, top_k=3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.top_k = top_k
        self.embedding_dim = embedding_dim
        print(f"Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.knowledge_captions = []

    def build_knowledge_base(self, embeddings: np.ndarray, captions: list):
        assert len(embeddings) == len(captions), "Embeddings and captions count must match."
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))
        self.knowledge_captions = captions
        print(f"Successfully indexed {self.index.ntotal} items in FAISS.")

    def retrieve_evidence(self, query_embedding: np.ndarray) -> list:
        q_copy = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q_copy)
        distances, indices = self.index.search(q_copy, self.top_k)
        return [self.knowledge_captions[i] for i in indices[0]]

    def generate_baseline(self, visual_description: str) -> str:
        prompt = f"Task: Generate an image caption for: {visual_description}\nCaption:"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=50, num_beams=3, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_rag(self, visual_description: str, retrieved_evidence: list) -> str:
        evidence_str = "\n".join([f"- {cap}" for cap in retrieved_evidence])
        prompt = (
            f"Task: Generate a precise and accurate image caption.\n"
            f"Visual Description: {visual_description}\n"
            f"Reference Evidence from Similar Images:\n{evidence_str}\n"
            f"Grounded Caption:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=50, num_beams=3, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
