
import torch

from pathlib import Path

from .dense_retrieval import DenseRetriever
from .sparse_retrieval import SparseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FEATURES_PATH = (
    BASE_DIR / "hybrid_features" / "train_hybrid_features.pt"
)


class HybridRetriever:

    def __init__(self, rrf_k=60):

        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()

        self.rrf_k = rrf_k

    def build_indexes(self):

        print("=" * 60)
        print("BUILDING HYBRID RETRIEVAL SYSTEM")
        print("=" * 60)

        # Load existing FAISS index
        index_path = (
            BASE_DIR / "retrieval" / "indexes" / "clip_train.index"
        )

        metadata_path = (
            BASE_DIR / "retrieval" / "indexes"
            / "clip_train_metadata.pt"
        )

        if index_path.exists() and metadata_path.exists():

            print("\nLoading existing Dense Retrieval index...")
            self.dense_retriever.load_index()

        else:

            print("\nFAISS index not found. Building it...")
            self.dense_retriever.build_index()

        # Build BM25 index
        print("\nBuilding Sparse Retrieval index...")
        self.sparse_retriever.build_index()

        print("\nHybrid indexes are ready!")

    def reciprocal_rank_fusion(
        self,
        dense_results,
        sparse_results,
        top_k=5
    ):

        fused_scores = {}
        image_information = {}

        # Process Dense Retrieval results
        for rank, result in enumerate(dense_results, start=1):

            image_id = result["image_id"]

            score = 1.0 / (self.rrf_k + rank)

            fused_scores[image_id] = (
                fused_scores.get(image_id, 0.0) + score
            )

            image_information[image_id] = {
                "image_id": image_id,
                "image_name": result["image_name"],
                "dense_similarity": result["similarity"],
                "sparse_score": None,
                "object_text": None
            }

        # Process Sparse Retrieval results
        for rank, result in enumerate(sparse_results, start=1):

            image_id = result["image_id"]

            score = 1.0 / (self.rrf_k + rank)

            fused_scores[image_id] = (
                fused_scores.get(image_id, 0.0) + score
            )

            if image_id not in image_information:

                image_information[image_id] = {
                    "image_id": image_id,
                    "image_name": result["image_name"],
                    "dense_similarity": None,
                    "sparse_score": result["score"],
                    "object_text": result["object_text"]
                }

            else:

                image_information[image_id]["sparse_score"] = (
                    result["score"]
                )

                image_information[image_id]["object_text"] = (
                    result["object_text"]
                )

        # Sort by final RRF score
        ranked_ids = sorted(
            fused_scores,
            key=lambda image_id: fused_scores[image_id],
            reverse=True
        )

        final_results = []

        for image_id in ranked_ids[:top_k]:

            result = image_information[image_id].copy()

            result["rrf_score"] = fused_scores[image_id]

            final_results.append(result)

        return final_results

    def search(
        self,
        query_embedding,
        query_text,
        top_k=5,
        candidate_k=50
    ):

        if not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        # Dense search using CLIP + FAISS
        dense_results = self.dense_retriever.search(
            query_embedding,
            top_k=candidate_k
        )

        # Sparse search using YOLO labels + BM25
        sparse_results = self.sparse_retriever.search(
            query_text,
            top_k=candidate_k
        )

        # Fuse both rankings using RRF
        hybrid_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            top_k=top_k
        )

        return hybrid_results


if __name__ == "__main__":

    retriever = HybridRetriever(rrf_k=60)

    # Build / load both indexes
    retriever.build_indexes()

    # Load Train features for a technical test
    data = torch.load(
        TRAIN_FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    # Use the first Train image as a test query
    query_embedding = (
        data["clip_embeddings"][0].numpy()
    )

    query_text = data["object_texts"][0]

    print("\n" + "=" * 60)
    print("TESTING HYBRID RETRIEVAL")
    print("=" * 60)

    print(f"\nQuery objects: {query_text}")

    results = retriever.search(
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=5,
        candidate_k=50
    )

    print("\nTop 5 Hybrid Retrieval Results:")

    for rank, result in enumerate(results, start=1):

        print(
            f"\n{rank}. {result['image_name']}"
            f"\n   Image ID: {result['image_id']}"
            f"\n   RRF Score: {result['rrf_score']:.6f}"
            f"\n   CLIP Similarity: {result['dense_similarity']}"
            f"\n   BM25 Score: {result['sparse_score']}"
            f"\n   Objects: {result['object_text']}"
        )
