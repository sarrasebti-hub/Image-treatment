
import re
import torch

from pathlib import Path
from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FEATURES_PATH = (
    BASE_DIR / "hybrid_features" / "train_hybrid_features.pt"
)


def tokenize(text):
    """Convert object text into searchable tokens."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class SparseRetriever:

    def __init__(self):
        self.bm25 = None
        self.image_ids = None
        self.image_names = None
        self.documents = None
        self.object_texts = None

    def build_index(self):

        print("=" * 60)
        print("BUILDING BM25 SPARSE RETRIEVAL INDEX")
        print("=" * 60)

        print("\nLoading Train hybrid features...")

        data = torch.load(
            TRAIN_FEATURES_PATH,
            map_location="cpu",
            weights_only=False
        )

        self.image_ids = data["image_ids"].tolist()
        self.image_names = data["image_names"]

        # Load original YOLO object descriptions
        self.object_texts = data["object_texts"]

        # Tokenize each document for BM25
        self.documents = [
            tokenize(text) for text in self.object_texts
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.documents)

        print("\nBM25 index successfully created!")
        print(f"Images indexed: {len(self.image_ids)}")
        print(f"Documents: {len(self.documents)}")

        empty_documents = sum(
            1 for document in self.documents if not document
        )

        print(f"Empty documents: {empty_documents}")

    def search(self, query, top_k=5):

        if self.bm25 is None:
            raise RuntimeError("Build the BM25 index first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError("Query contains no searchable words.")

        # Calculate BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort results by descending BM25 score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True
        )

        results = []

        for idx in ranked_indices[:top_k]:

            results.append({
                "image_id": int(self.image_ids[idx]),
                "image_name": self.image_names[idx],
                "score": float(scores[idx]),
                "object_text": self.object_texts[idx]
            })

        return results


if __name__ == "__main__":

    retriever = SparseRetriever()

    # Build BM25 index
    retriever.build_index()

    # Example query based on YOLO object labels
    query = "person car bicycle traffic light"

    # Search
    results = retriever.search(
        query,
        top_k=5
    )

    print(f"\nQuery: {query}")
    print("\nTop 5 BM25 results:")

    for rank, result in enumerate(results, start=1):

        print(
            f"\n{rank}. {result['image_name']}"
            f"\n   Image ID: {result['image_id']}"
            f"\n   BM25 Score: {result['score']:.4f}"
            f"\n   Objects: {result['object_text']}"
        )
