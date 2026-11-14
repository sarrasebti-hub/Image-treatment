
import sys
import json
import csv

from pathlib import Path

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

from rouge_score import rouge_scorer


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_IMAGES_DIR = BASE_DIR / "dataset" / "images_val"

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# Original predictions: used to select the same images
OLD_PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

# New V2 predictions
V2_PREDICTIONS_PATH = (
    RESULTS_DIR / "hybrid_rag_v2_predictions.json"
)

V2_METRICS_PATH = (
    RESULTS_DIR / "hybrid_rag_v2_metrics.json"
)

V2_CSV_PATH = (
    RESULTS_DIR / "hybrid_rag_v2_results.csv"
)

INVALID_CAPTIONS = [
    "use retrieved captions only as supporting context."
]


# ============================================================
# IMPORT GENERATION SYSTEM
# ============================================================

sys.path.insert(0, str(BASE_DIR / "generation"))

from caption_generation import CaptionGenerationSystem


# ============================================================
# LOAD THE SAME 99 IMAGES AS THE OLD EVALUATION
# ============================================================

def load_comparison_images():

    print("\nLoading original evaluation predictions...")

    if not OLD_PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            f"Original predictions not found: {OLD_PREDICTIONS_PATH}"
        )

    with open(
        OLD_PREDICTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        old_predictions = json.load(file)

    selected_images = {}

    for image_id, item in old_predictions.items():

        caption = item.get(
            "generated_caption",
            ""
        ).strip()

        references = item.get(
            "references",
            []
        )

        # Exclude the same invalid prediction as the old evaluation
        if caption.lower() in INVALID_CAPTIONS:
            continue

        if not references:
            continue

        image_name = item.get(
            "image_name",
            f"{int(image_id):012d}.jpg"
        )

        image_path = TEST_IMAGES_DIR / image_name

        if not image_path.exists():
            continue

        selected_images[str(image_id)] = {

            "image_id": int(image_id),

            "image_name": image_name,

            "image_path": str(image_path),

            # Reuse exactly the same references
            "references": references
        }

    print(
        f"Images selected for comparison: {len(selected_images)}"
    )

    if not selected_images:

        raise ValueError(
            "No valid comparison images found."
        )

    return selected_images


# ============================================================
# LOAD EXISTING V2 PREDICTIONS
# ============================================================

def load_v2_predictions():

    if not V2_PREDICTIONS_PATH.exists():
        return {}

    with open(
        V2_PREDICTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        predictions = json.load(file)

    print(
        f"Resuming V2 evaluation: "
        f"{len(predictions)} saved predictions."
    )

    return predictions


# ============================================================
# SAVE V2 PREDICTIONS
# ============================================================

def save_v2_predictions(predictions):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        V2_PREDICTIONS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            predictions,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# GENERATE V2 CAPTIONS
# ============================================================

def generate_v2_predictions(images):

    print("\n" + "=" * 65)
    print("GENERATING HYBRID RAG V2 PREDICTIONS")
    print("=" * 65)

    predictions = load_v2_predictions()

    system = CaptionGenerationSystem()

    # Load models and retrieval indexes only once
    system.initialize()

    total = len(images)

    for position, (image_id, image_data) in enumerate(
        images.items(),
        start=1
    ):

        # Resume interrupted evaluation
        if image_id in predictions:

            print(
                f"\n[{position}/{total}] "
                f"Image {image_id} already processed."
            )

            continue

        print("\n" + "#" * 65)
        print(f"PROCESSING IMAGE {position}/{total}")
        print(f"Image: {image_data['image_name']}")
        print("#" * 65)

        try:

            result = system.generate(
                image_data["image_path"]
            )

            predictions[image_id] = {

                "image_id": image_data["image_id"],

                "image_name": image_data["image_name"],

                "generated_caption": result["final_caption"],

                "references": image_data["references"],

                "initial_caption": result["initial_caption"],

                "used_fallback": result["used_fallback"]

            }

            save_v2_predictions(predictions)

            print("\nV2 prediction saved successfully!")

        except Exception as error:

            print(
                f"\nError processing image {image_id}: {error}"
            )

            save_v2_predictions(predictions)

            continue

    print("\nV2 caption generation completed.")

    print(
        f"Saved predictions: {len(predictions)}"
    )

    return predictions


# ============================================================
# FILTER AND ALIGN PREDICTIONS
# ============================================================

def prepare_valid_predictions(predictions, images):

    valid_predictions = {}

    for image_id, image_data in images.items():

        if image_id not in predictions:

            print(
                f"Missing prediction for image {image_id}"
            )

            continue

        item = predictions[image_id]

        caption = item.get(
            "generated_caption",
            ""
        ).strip()

        if not caption:

            print(
                f"Empty caption for image {image_id}"
            )

            continue

        if caption.lower() in INVALID_CAPTIONS:

            print(
                f"Invalid caption for image {image_id}"
            )

            continue

        # Ensure that references are exactly the original references
        item["references"] = image_data["references"]

        valid_predictions[image_id] = item

    print("\n" + "=" * 65)

    print(
        f"Valid V2 predictions: {len(valid_predictions)}"
    )

    print(
        f"Missing or invalid: "
        f"{len(images) - len(valid_predictions)}"
    )

    return valid_predictions


# ============================================================
# PREPARE COCO DATA
# ============================================================

def prepare_coco_data(predictions):

    ground_truths = {}
    results = {}

    for image_id, item in predictions.items():

        numeric_id = int(image_id)

        ground_truths[numeric_id] = [
            {"caption": caption}
            for caption in item["references"]
        ]

        results[numeric_id] = [
            {"caption": item["generated_caption"]}
        ]

    return ground_truths, results


# ============================================================
# BLEU AND CIDEr
# ============================================================

def calculate_coco_metrics(ground_truths, results):

    metrics = {}

    print("\nTokenizing captions with PTBTokenizer...")

    tokenizer = PTBTokenizer()

    ground_truths = tokenizer.tokenize(
        ground_truths
    )

    results = tokenizer.tokenize(
        results
    )

    print("Tokenization completed.")

    # BLEU
    print("\nCalculating BLEU...")

    bleu_scorer = Bleu(4)

    bleu_scores, _ = bleu_scorer.compute_score(
        ground_truths,
        results
    )

    metrics["BLEU-1"] = float(bleu_scores[0])

    metrics["BLEU-4"] = float(bleu_scores[3])

    print("BLEU calculated successfully.")

    # CIDEr
    print("\nCalculating CIDEr...")

    cider_scorer = Cider()

    cider_score, _ = cider_scorer.compute_score(
        ground_truths,
        results
    )

    metrics["CIDEr"] = float(cider_score)

    print("CIDEr calculated successfully.")

    return metrics


# ============================================================
# ROUGE-L
# ============================================================

def calculate_rouge_l(predictions):

    print("\nCalculating ROUGE-L...")

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    scores = []

    for item in predictions.values():

        generated = item["generated_caption"]

        references = item["references"]

        reference_scores = []

        for reference in references:

            score = scorer.score(
                reference,
                generated
            )

            reference_scores.append(
                score["rougeL"].fmeasure
            )

        if reference_scores:

            scores.append(
                max(reference_scores)
            )

    return (
        sum(scores) / len(scores)
        if scores else 0.0
    )


# ============================================================
# FALLBACK STATISTICS
# ============================================================

def calculate_fallback_statistics(predictions):

    total = len(predictions)

    fallback_count = sum(
        1
        for item in predictions.values()
        if item.get("used_fallback", False)
    )

    generated_count = total - fallback_count

    fallback_rate = (
        (fallback_count / total) * 100
        if total else 0.0
    )

    return {

        "fallback_count": fallback_count,

        "flan_generated_count": generated_count,

        "fallback_rate_percent": round(
            fallback_rate,
            2
        )
    }


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        V2_METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        V2_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Score"
        ])

        for name, score in metrics.items():

            writer.writerow([
                name,
                score
            ])

    print("\nMetrics saved successfully.")

    print(f"JSON: {V2_METRICS_PATH}")

    print(f"CSV: {V2_CSV_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # STEP 1: Select original comparison images
    images = load_comparison_images()

    # STEP 2: Generate new V2 predictions
    predictions = generate_v2_predictions(
        images
    )

    # STEP 3: Keep valid predictions
    predictions = prepare_valid_predictions(
        predictions,
        images
    )

    if not predictions:

        raise ValueError(
            "No valid V2 predictions available."
        )

    # STEP 4: Prepare COCO data
    ground_truths, results = prepare_coco_data(
        predictions
    )

    print("\n" + "=" * 65)
    print("STARTING HYBRID RAG V2 EVALUATION")
    print("=" * 65)

    # STEP 5: Calculate BLEU and CIDEr
    metrics = calculate_coco_metrics(
        ground_truths,
        results
    )

    # STEP 6: Calculate ROUGE-L
    metrics["ROUGE-L"] = calculate_rouge_l(
        predictions
    )

    # STEP 7: Fallback statistics
    fallback_stats = calculate_fallback_statistics(
        predictions
    )

    metrics.update(
        fallback_stats
    )

    metrics["number_of_images"] = len(
        predictions
    )

    # STEP 8: Save results
    save_metrics(metrics)

    # STEP 9: Display results
    print("\n" + "=" * 65)
    print("HYBRID RAG V2 FINAL RESULTS")
    print("=" * 65)

    for metric, score in metrics.items():

        if isinstance(score, float):

            print(f"{metric}: {score:.4f}")

        else:

            print(f"{metric}: {score}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":

    main()

