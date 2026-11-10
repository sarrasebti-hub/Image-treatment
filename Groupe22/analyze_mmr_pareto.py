import numpy as np

results = {
    "CLIP Reranking": {
        "relevance": 0.3235,
        "diversity": 0.2295
    },
    "MMR Lambda 0.7": {
        "relevance": 0.2966,
        "diversity": 0.4415
    },
    "MMR Lambda 0.9": {
        "relevance": 0.3208,
        "diversity": 0.2956
    }
}

print("========== PARETO ANALYSIS ==========")

for method, metrics in results.items():
    print(f"\n{method}")
    print("Relevance:", metrics["relevance"])
    print("Diversity:", metrics["diversity"])

print("\n========== DOMINANCE ANALYSIS ==========")

for a, b in [
    ("CLIP Reranking", "MMR Lambda 0.7"),
    ("CLIP Reranking", "MMR Lambda 0.9"),
    ("MMR Lambda 0.7", "MMR Lambda 0.9")
]:

    A = results[a]
    B = results[b]

    dominates = (
        A["relevance"] >= B["relevance"]
        and A["diversity"] >= B["diversity"]
        and (
            A["relevance"] > B["relevance"]
            or A["diversity"] > B["diversity"]
        )
    )

    print(f"\n{a} dominates {b}: {dominates}")

print("\n=====================================")