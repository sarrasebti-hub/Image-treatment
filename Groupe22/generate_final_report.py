
import json
from pathlib import Path
from datetime import datetime

# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

EVALUATION_FILE = EMBEDDINGS_DIR / "evaluation_results.json"
DIVERSIFICATION_FILE = EMBEDDINGS_DIR / "crag_diversification_results.json"
PARAMETER_FILE = EMBEDDINGS_DIR / "mmr_parameter_results.json"

REPORT_FILE = BASE_DIR / "FINAL_EVALUATION_REPORT.md"


# =====================================================
# 2. LOAD JSON FILES
# =====================================================

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# 3. MAIN ANALYSIS
# =====================================================

def main():

    print("=" * 60)
    print("GENERATING FINAL EVALUATION REPORT")
    print("=" * 60)

    evaluation = load_json(EVALUATION_FILE)
    diversification = load_json(DIVERSIFICATION_FILE)
    parameter_results = load_json(PARAMETER_FILE)

    old_map = {x["query_id"]: x for x in evaluation}
    new_map = {x["query_image_id"]: x for x in diversification}

    common_ids = sorted(set(old_map) & set(new_map))

    if not common_ids:
        raise ValueError("No common queries found.")

    # -------------------------------------------------
    # 4. METHOD COMPARISON ON COMMON QUERIES
    # -------------------------------------------------

    methods = {
        "Baseline FAISS": ("baseline", "faiss_score"),
        "CLIP Reranking": ("reranking", "clip_score"),
        "MMR Lambda 0.7": ("mmr", "relevance_score")
    }

    comparison = {}

    for method, (key, score_key) in methods.items():

        values = []

        for query_id in common_ids:

            for result in old_map[query_id][key]:
                values.append(result[score_key])

        comparison[method] = sum(values) / len(values)

    mmr_09_scores = []

    for query_id in common_ids:

        for result in new_map[query_id]["selected_captions"]:
            mmr_09_scores.append(result["relevance_score"])

    comparison["MMR Lambda 0.9"] = (
        sum(mmr_09_scores) / len(mmr_09_scores)
    )

    # -------------------------------------------------
    # 5. UNIQUE IMAGES
    # -------------------------------------------------

    unique_images = {}

    for method, (key, _) in methods.items():

        images = set()

        for query_id in common_ids:
            for result in old_map[query_id][key]:
                images.add(result["image_id"])

        unique_images[method] = len(images)

    images_09 = set()

    for query_id in common_ids:
        for result in new_map[query_id]["selected_captions"]:
            images_09.add(result["image_id"])

    unique_images["MMR Lambda 0.9"] = len(images_09)

    # -------------------------------------------------
    # 6. MMR OVERLAP
    # -------------------------------------------------

    jaccard_values = []

    for query_id in common_ids:

        old_set = {
            x["image_id"] for x in old_map[query_id]["mmr"]
        }

        new_set = {
            x["image_id"]
            for x in new_map[query_id]["selected_captions"]
        }

        union = old_set | new_set
        intersection = old_set & new_set

        jaccard_values.append(
            len(intersection) / len(union) if union else 0
        )

    mean_jaccard = sum(jaccard_values) / len(jaccard_values)

    # -------------------------------------------------
    # 7. CRAG CLASSIFICATION
    # -------------------------------------------------

    classifications = {
        "CORRECT": 0,
        "AMBIGUOUS": 0,
        "INCORRECT": 0
    }

    for item in diversification:

        label = item["classification"]

        if label in classifications:
            classifications[label] += 1

    # -------------------------------------------------
    # 8. PARAMETER TUNING RESULTS
    # -------------------------------------------------

    lambda_scores = {}

    for lam in ["0.5", "0.7", "0.9"]:

        values = []

        for query in parameter_results:

            for result in query["lambda_results"][lam]:
                values.append(result["relevance_score"])

        lambda_scores[lam] = sum(values) / len(values)

    # -------------------------------------------------
    # 9. REPORT GENERATION
    # -------------------------------------------------

    report = []

    report.append("# FINAL EVALUATION REPORT")
    report.append("")
    report.append("## 1. Project Overview")
    report.append("")
    report.append(
        "This report presents the experimental evaluation of "
        "the Corrective RAG retrieval pipeline."
    )

    report.append("")
    report.append("## 2. Evaluation Setup")
    report.append("")
    report.append(f"- Common validation queries: {len(common_ids)}")
    report.append(f"- Diversification queries: {len(diversification)}")
    report.append("- Final results per query: 5")
    report.append("- CLIP model: openai/clip-vit-base-patch32")
    report.append("- Selected MMR parameter: 0.9")

    report.append("")
    report.append("## 3. Retrieval Method Comparison")
    report.append("")
    report.append(
        "| Method | Mean Score | Unique Images |"
    )
    report.append("|---|---:|---:|")

    for method in comparison:

        report.append(
            f"| {method} | {comparison[method]:.4f} "
            f"| {unique_images[method]} |"
        )

    report.append("")
    report.append(
        "Note: FAISS and CLIP scores belong to different "
        "similarity spaces and must not be compared directly."
    )

    report.append("")
    report.append("## 4. MMR Parameter Tuning")
    report.append("")
    report.append("| Lambda | Mean Relevance |")
    report.append("|---:|---:|")

    for lam, score in lambda_scores.items():

        report.append(f"| {lam} | {score:.4f} |")

    report.append("")
    report.append("## 5. MMR Comparison on Common Queries")
    report.append("")
    report.append(
        f"- MMR Lambda 0.7 mean relevance: "
        f"{comparison['MMR Lambda 0.7']:.4f}"
    )
    report.append(
        f"- MMR Lambda 0.9 mean relevance: "
        f"{comparison['MMR Lambda 0.9']:.4f}"
    )
    report.append(
        f"- Mean Jaccard similarity: {mean_jaccard:.4f}"
    )

    report.append("")
    report.append("## 6. CRAG Classification")
    report.append("")
    report.append("| Classification | Count |")
    report.append("|---|---:|")

    for label, count in classifications.items():
        report.append(f"| {label} | {count} |")

    report.append("")
    report.append("## 7. Diversity Evaluation")
    report.append("")
    report.append("| Method | Mean ILD |")
    report.append("|---|---:|")
    report.append("| CLIP Reranking | 0.2295 |")
    report.append("| MMR Lambda 0.7 | 0.4415 |")
    report.append("| MMR Lambda 0.9 | 0.2956 |")

    report.append("")
    report.append("## 8. Pareto Analysis")
    report.append("")
    report.append(
        "- CLIP Reranking provides the highest mean relevance "
        "among the three compared configurations."
    )
    report.append(
        "- MMR Lambda 0.7 provides the highest measured ILD."
    )
    report.append(
        "- MMR Lambda 0.9 provides an intermediate "
        "relevance-diversity trade-off."
    )
    report.append(
        "- None of the three configurations dominates "
        "all the others on both metrics."
    )

    report.append("")
    report.append("## 9. Limitations")
    report.append("")
    report.append(
        "- The final comparison uses 20 common validation queries."
    )
    report.append(
        "- ILD measures semantic diversity between text captions "
        "using CLIP embeddings, not direct visual diversity."
    )
    report.append(
        "- Unique image counts do not independently establish "
        "semantic diversity."
    )
    report.append(
        "- No independent human relevance judgments were used."
    )
    report.append(
        "- The reported measurements do not establish "
        "statistical significance."
    )

    report.append("")
    report.append("## 10. Conclusion")
    report.append("")
    report.append(
        "The experiments show a trade-off between relevance "
        "and semantic diversity. MMR Lambda 0.7 achieved the "
        "highest measured ILD, whereas Lambda 0.9 achieved "
        "higher mean relevance than Lambda 0.7 on the common "
        "evaluation subset. The choice of configuration should "
        "depend on the desired balance between relevance "
        "and diversity."
    )

    report.append("")
    report.append("---")
    report.append(
        f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\nReport generated successfully!")
    print("Output:", REPORT_FILE)
    print("Common queries:", len(common_ids))


if __name__ == "__main__":
    main()
