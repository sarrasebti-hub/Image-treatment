
import torch
import faiss
import numpy as np

from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FEATURES_PATH = (
    BASE_DIR / "hybrid_features" / "train_hybrid_features.pt"
)

INDEX_DIR = BASE_DIR / "retrieval" / "indexes"
INDEX_PATH = INDEX_DIR / "clip_train.index"

METADATA_PATH = INDEX_DIR / "clip_train_metadata.pt"


class DenseRetriever:

    def __init__(self):
        self.index = None
        self.image_ids = None
        self.image_names = None

    def build_index(self):

        print("=" * 60)
        print("BUILDING FAISS DENSE RETRIEVAL INDEX")
        print("=" * 60)

        # Load Train features
        print("\nLoading Train hybrid features...")

        data = torch.load(
            TRAIN_FEATURES_PATH,
            map_location="cpu",
            weights_only=False
        )

        embeddings = data["clip_embeddings"].numpy().astype(
            np.float32
        )

        self.image_ids = data["image_ids"].numpy()
        self.image_names = data["image_names"]

        # Validate embeddings
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or Infinity.")

        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Embeddings are not normalized.")

        # Create FAISS index
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        # Save index and metadata
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(INDEX_PATH))

        metadata = {
            "image_ids": self.image_ids,
            "image_names": self.image_names,
            "dimension": dimension,
            "number_of_images": len(self.image_ids)
        }

        torch.save(metadata, METADATA_PATH)

        print("\nIndex successfully created!")
        print(f"Images indexed: {self.index.ntotal}")
        print(f"Embedding dimension: {dimension}")
        print(f"Index path: {INDEX_PATH}")

    def load_index(self):

        print("Loading FAISS index...")

        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run build_index() first."
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        metadata = torch.load(
            METADATA_PATH,
            map_location="cpu",
            weights_only=False
        )

        self.image_ids = metadata["image_ids"]
        self.image_names = metadata["image_names"]

        print(f"Loaded {self.index.ntotal} images.")

    def search(self, query_embedding, top_k=5):

        if self.index is None:
            raise RuntimeError("Load or build the index first.")

        query = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension mismatch.")

        if not np.isfinite(query).all():
            raise ValueError("Query contains NaN or Infinity.")

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("Query embedding has zero norm.")

        # Normalize query for cosine similarity
        query = query / norm

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            results.append({
                "image_id": int(self.image_ids[idx]),
                "image_name": self.image_names[idx],
                "similarity": float(score)
            })

        return results


if __name__ == "__main__":

    retriever = DenseRetriever()

    retriever.build_index()

    # Test search using the first Train embedding
    data = torch.load(
        TRAIN_FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    query_embedding = data["clip_embeddings"][0].numpy()

    results = retriever.search(
        query_embedding,
        top_k=5
    )

    print("\nTop 5 similar images:")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['image_name']} | "
            f"Similarity: {result['similarity']:.4f}"
        )
