
import json
import sys
from pathlib import Path

import torch
import pandas as pd

# Allow imports from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.dense_retrieval import DenseRetriever
from retrieval.sparse_retrieval import SparseRetriever
from retrieval.hybrid_retrieval import HybridRetriever


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRAIN_PATH = BASE_DIR / "hybrid_features" / "train_hybrid_features.pt"
VAL_PATH = BASE_DIR / "hybrid_features" / "validation_hybrid_features.pt"

OUTPUT_DIR = BASE_DIR / "evaluation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RELEVANCE_THRESHOLD = 0.5
TOP_K = 10
CANDIDATE_K = 100


# --------------------------------------------------
# OBJECT LABEL PROCESSING
# --------------------------------------------------

def extract_labels(object_text):
    """Convert YOLO object text into a set of unique labels."""

    if not object_text:
        return set()

    return {
        label.strip().lower()
        for label in object_text.split(",")
        if label.strip()
    }


def jaccard_similarity(set_a, set_b):
    """Calculate Jaccard similarity between two object sets."""

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union else 0.0


# --------------------------------------------------
# GROUND TRUTH (PSEUDO-RELEVANCE)
# --------------------------------------------------

def build_relevance_sets(train_data, val_data):

    train_ids = train_data["image_ids"].tolist()
    train_texts = train_data["object_texts"]

    val_texts = val_data["object_texts"]

    train_labels = [
        extract_labels(text)
        for text in train_texts
    ]

    relevance_sets = []
    skipped_no_objects = 0
    skipped_no_relevant = 0

    for query_text in val_texts:

        query_labels = extract_labels(query_text)

        if not query_labels:
            relevance_sets.append(None)
            skipped_no_objects += 1
            continue

        relevant_ids = set()

        for train_id, candidate_labels in zip(
            train_ids, train_labels
        ):

            similarity = jaccard_similarity(
                query_labels,
                candidate_labels
            )

            if similarity >= RELEVANCE_THRESHOLD:
                relevant_ids.add(int(train_id))

        if not relevant_ids:
            relevance_sets.append(None)
            skipped_no_relevant += 1
        else:
            relevance_sets.append(relevant_ids)

    return (
        relevance_sets,
        skipped_no_objects,
        skipped_no_relevant
    )


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(results, relevant_ids, k):

    retrieved_ids = [
        int(result["image_id"])
        for result in results[:k]
    ]

    relevant_retrieved = sum(
        1 for image_id in retrieved_ids
        if image_id in relevant_ids
    )

    precision = relevant_retrieved / k

    recall = relevant_retrieved / len(relevant_ids)

    reciprocal_rank = 0.0

    for rank, image_id in enumerate(retrieved_ids, start=1):

        if image_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return precision, recall, reciprocal_rank


def evaluate_method(name, search_function, relevance_sets):

    totals = {
        "P@5": 0.0,
        "P@10": 0.0,
        "R@5": 0.0,
        "R@10": 0.0,
        "MRR@5": 0.0,
        "MRR@10": 0.0
    }

    evaluated = 0

    for query_index, relevant_ids in enumerate(relevance_sets):

        if not relevant_ids:
            continue

        results = search_function(query_index)

        p5, r5, mrr5 = calculate_metrics(
            results, relevant_ids, 5
        )

        p10, r10, mrr10 = calculate_metrics(
            results, relevant_ids, 10
        )

        totals["P@5"] += p5
        totals["R@5"] += r5
        totals["MRR@5"] += mrr5

        totals["P@10"] += p10
        totals["R@10"] += r10
        totals["MRR@10"] += mrr10

        evaluated += 1

        if (evaluated % 20) == 0:
            print(f"{name}: {evaluated} queries evaluated")

    if evaluated == 0:
        raise RuntimeError(
            "No validation queries have relevant training images. "
            "Check the relevance threshold."
        )

    return {
        metric: round(value / evaluated, 4)
        for metric, value in totals.items()
    }, evaluated


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("RETRIEVAL EVALUATION: DENSE vs SPARSE vs HYBRID")
    print("=" * 65)

    print("\nLoading Train and Validation features...")

    train_data = torch.load(
        TRAIN_PATH,
        map_location="cpu",
        weights_only=False
    )

    val_data = torch.load(
        VAL_PATH,
        map_location="cpu",
        weights_only=False
    )

    val_embeddings = val_data["clip_embeddings"]
    val_texts = val_data["object_texts"]

    print(f"Train images: {len(train_data['image_ids'])}")
    print(f"Validation images: {len(val_data['image_ids'])}")

    # Build pseudo-relevance sets
    print("\nBuilding pseudo-relevance sets...")

    (
        relevance_sets,
        skipped_no_objects,
        skipped_no_relevant
    ) = build_relevance_sets(train_data, val_data)

    evaluated_queries = sum(
        1 for item in relevance_sets if item
    )

    print(f"Evaluated queries: {evaluated_queries}")
    print(f"Skipped (no YOLO objects): {skipped_no_objects}")
    print(f"Skipped (no relevant image): {skipped_no_relevant}")

    if evaluated_queries == 0:
        raise RuntimeError("No queries available for evaluation.")

    # Initialize Dense
    print("\nLoading Dense Retriever...")

    dense = DenseRetriever()
    dense.load_index()

    # Initialize Sparse
    print("\nBuilding Sparse Retriever...")

    sparse = SparseRetriever()
    sparse.build_index()

    # Initialize Hybrid
    print("\nInitializing Hybrid Retriever...")

    hybrid = HybridRetriever()
    hybrid.build_indexes()

    # Search functions
    def dense_search(i):
        return dense.search(
            val_embeddings[i].numpy(),
            top_k=TOP_K
        )

    def sparse_search(i):
        return sparse.search(
            val_texts[i],
            top_k=TOP_K
        )

    def hybrid_search(i):
        return hybrid.search(
            val_embeddings[i].numpy(),
            val_texts[i],
            top_k=TOP_K,
            candidate_k=CANDIDATE_K
        )

    methods = [
        ("Dense", dense_search),
        ("Sparse", sparse_search),
        ("Hybrid", hybrid_search)
    ]

    all_results = {}

    print("\nStarting evaluation...")

    for name, search_function in methods:

        print("\n" + "-" * 65)
        print(f"Evaluating {name} Retrieval")
        print("-" * 65)

        metrics, count = evaluate_method(
            name,
            search_function,
            relevance_sets
        )

        all_results[name] = metrics

        print(f"\n{name} Results:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Save JSON
    report = {
        "evaluation_method": (
            "YOLO object-label Jaccard similarity "
            "used as pseudo-relevance"
        ),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "total_validation_queries": len(val_texts),
        "evaluated_queries": evaluated_queries,
        "skipped_no_objects": skipped_no_objects,
        "skipped_no_relevant_images": skipped_no_relevant,
        "metrics": all_results
    }

    json_path = OUTPUT_DIR / "retrieval_metrics.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Save CSV
    df = pd.DataFrame.from_dict(
        all_results,
        orient="index"
    )

    df.index.name = "Method"

    csv_path = OUTPUT_DIR / "retrieval_metrics.csv"
    df.to_csv(csv_path)

    # Display comparison
    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 65)

    print(df.to_string())

    print("\nSaved files:")
    print(json_path)
    print(csv_path)

    print("\nRetrieval evaluation completed successfully!")


if __name__ == "__main__":
    main()
