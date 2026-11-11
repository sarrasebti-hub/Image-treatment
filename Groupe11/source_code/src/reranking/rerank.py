import torch
import numpy as np
from sentence_transformers import CrossEncoder

class RAG_Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Loading Cross-Encoder: {model_name}...")
        # This is a small, fast model that looks at the Query and the Document together
        self.model = CrossEncoder(model_name)

    def rerank(self, query_text, retrieved_captions, top_k=5):
        """
        query_text: The text we are searching for (e.g., "a dog in the park")
        retrieved_captions: A list of texts (the 20 candidates from FAISS)
        top_k: How many we want to keep (our target is 5)
        """
        # 1. Create pairs of [Query, Candidate]
        pairs = [[query_text, caption] for caption in retrieved_captions]
        
        # 2. Get scores from the Cross-Encoder
        scores = self.model.predict(pairs)
        
        # 3. Sort the captions based on the new scores (highest first)
        ranked_results = sorted(zip(scores, retrieved_captions), reverse=True)
        
        # 4. Return only the Top K
        return ranked_results[:top_k]

# --- Testing the Reranker ---
if __name__ == "__main__":
    # Fake data to test
    query = "a black dog running on the grass"
    candidates = [
        "a brown cat sitting on a couch",         # Bad
        "a black dog runs across the green lawn", # Perfect
        "a white dog sleeping",                   # Okay
        "a person walking a dog",                 # Okay
        "a red car parked outside"                # Terrible
    ]
    
    reranker = RAG_Reranker()
    best_5 = reranker.rerank(query, candidates, top_k=2)
    
    print("\n--- Results ---")
    print(f"Query: {query}")
    for score, text in best_5:
        print(f"Score: {score:.4f} | Caption: {text}")