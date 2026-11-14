
import sys
import json
import csv

from pathlib import Path
from collections import defaultdict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

from rouge_score import rouge_scorer


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_IMAGES_DIR = BASE_DIR / "dataset" / "images_val"

TEST_CAPTIONS_PATH = (
    BASE_DIR / "dataset" / "processed" / "test_captions.json"
)

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"

CSV_PATH = RESULTS_DIR / "evaluation_results.csv"

NUMBER_OF_IMAGES = 100


# ============================================================
# IMPORT GENERATION SYSTEM
# ============================================================

sys.path.insert(0, str(BASE_DIR / "generation"))

from caption_generation import CaptionGenerationSystem


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_data():

    print("\nLoading Test Dataset...")

    with open(TEST_CAPTIONS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    references = defaultdict(list)

    for annotation in data["annotations"]:

        image_id = int(annotation["image_id"])
        caption = annotation["caption"]

        references[image_id].append(caption)

    images = []

    for image in data["images"]:

        image_id = int(image["id"])

        image_name = image.get(
            "file_name",
            f"{image_id:012d}.jpg"
        )

        image_path = TEST_IMAGES_DIR / image_name

        if not image_path.exists():
            image_path = TEST_IMAGES_DIR / f"{image_id:012d}.jpg"

        if not image_path.exists():
            continue

        if image_id not in references:
            continue

        images.append({
            "image_id": image_id,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "references": references[image_id]
        })

    images.sort(key=lambda item: item["image_id"])

    selected_images = images[:NUMBER_OF_IMAGES]

    print(f"Available test images: {len(images)}")
    print(f"Selected images: {len(selected_images)}")

    if not selected_images:
        raise ValueError(
            "No valid test images found. Check dataset paths."
        )

    return selected_images


# ============================================================
# LOAD PREVIOUS PREDICTIONS
# ============================================================

def load_previous_predictions():

    if not PREDICTIONS_PATH.exists():
        return {}

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as file:
        predictions = json.load(file)

    print(
        f"Resuming from {len(predictions)} saved predictions."
    )

    return predictions


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(predictions):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        PREDICTIONS_PATH,
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
# GENERATE CAPTIONS FOR TEST IMAGES
# ============================================================

def generate_predictions(images):

    print("\n" + "=" * 65)
    print("STARTING TEST CAPTION GENERATION")
    print("=" * 65)

    predictions = load_previous_predictions()

    system = CaptionGenerationSystem()

    system.initialize()

    total = len(images)

    for position, image_data in enumerate(images, start=1):

        image_id = str(image_data["image_id"])

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

                "references": image_data["references"]

            }

            save_predictions(predictions)

            print("\nSaved prediction successfully!")

        except Exception as error:

            print(
                f"\nError processing image {image_id}: {error}"
            )

            save_predictions(predictions)

            continue

    print("\nCaption generation stage completed.")

    print(
        f"Successfully generated: {len(predictions)} images."
    )

    return predictions


# ============================================================
# PREPARE COCO EVALUATION DATA
# ============================================================

def prepare_coco_data(predictions):

    ground_truths = {}
    results = {}

    for image_id, item in predictions.items():

        references = [
            {"caption": caption}
            for caption in item["references"]
        ]

        generated = [
            {"caption": item["generated_caption"]}
        ]

        numeric_id = int(image_id)

        ground_truths[numeric_id] = references
        results[numeric_id] = generated

    return ground_truths, results


# ============================================================
# FILTER INVALID PREDICTIONS
# ============================================================

def filter_invalid_predictions(predictions):

    invalid_phrases = [
        "use retrieved captions only as supporting context."
    ]

    valid_predictions = {}

    for image_id, item in predictions.items():

        caption = item.get(
            "generated_caption",
            ""
        ).strip()

        if not caption:

            print(
                f"Excluding empty caption for image {image_id}"
            )

            continue

        if caption.lower() in invalid_phrases:

            print(
                f"Excluding invalid caption for image {image_id}"
            )

            continue

        if not item.get("references"):

            print(
                f"Excluding image without references: {image_id}"
            )

            continue

        valid_predictions[image_id] = item

    print(
        f"\nValid predictions: {len(valid_predictions)}"
    )

    print(
        f"Excluded predictions: "
        f"{len(predictions) - len(valid_predictions)}"
    )

    return valid_predictions


# ============================================================
# ROUGE-L EVALUATION
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
            scores.append(max(reference_scores))

    final_score = (
        sum(scores) / len(scores)
        if scores else 0.0
    )

    print("ROUGE-L calculated successfully.")

    return final_score


# ============================================================
# COCO METRICS
# ============================================================

def calculate_coco_metrics(ground_truths, results):

    metrics = {}

    # ----------------------------------------
    # TOKENIZATION
    # ----------------------------------------

    print("\nTokenizing captions with PTBTokenizer...")

    tokenizer = PTBTokenizer()

    ground_truths = tokenizer.tokenize(ground_truths)

    results = tokenizer.tokenize(results)

    print("Tokenization completed.")

    # ----------------------------------------
    # BLEU
    # ----------------------------------------

    print("\nCalculating BLEU-1 and BLEU-4...")

    bleu_scorer = Bleu(4)

    bleu_scores, _ = bleu_scorer.compute_score(
        ground_truths,
        results
    )

    metrics["BLEU-1"] = float(bleu_scores[0])

    metrics["BLEU-4"] = float(bleu_scores[3])

    print("BLEU calculated successfully.")

    # ----------------------------------------
    # METEOR
    # ----------------------------------------

    print("\nMETEOR temporarily skipped.")

    # ----------------------------------------
    # CIDEr
    # ----------------------------------------

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
# SAVE METRICS
# ============================================================

def save_metrics(metrics, number_of_images):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics["number_of_images"] = number_of_images

    # Save JSON

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save CSV

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Score"])

        for name, score in metrics.items():

            writer.writerow([name, score])

    print("\nMetrics saved successfully.")

    print(f"JSON: {METRICS_PATH}")

    print(f"CSV: {CSV_PATH}")


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load existing predictions

    if not PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            "predictions.json not found. "
            "Run caption generation first."
        )

    print("\nLoading saved predictions...")

    with open(
        PREDICTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        predictions = json.load(file)

    print(
        f"Loaded {len(predictions)} saved predictions."
    )

    # Filter invalid captions

    predictions = filter_invalid_predictions(
        predictions
    )

    if not predictions:

        raise ValueError(
            "No valid predictions available."
        )

    # Prepare evaluation data

    ground_truths, results = prepare_coco_data(
        predictions
    )

    print("\n" + "=" * 65)
    print("STARTING METRICS EVALUATION")
    print("=" * 65)

    # Calculate BLEU and CIDEr

    metrics = calculate_coco_metrics(
        ground_truths,
        results
    )

    # Calculate ROUGE-L

    metrics["ROUGE-L"] = calculate_rouge_l(
        predictions
    )

    # Save metrics

    save_metrics(
        metrics,
        len(predictions)
    )

    # Display results

    print("\n" + "=" * 65)
    print("FINAL EVALUATION RESULTS")
    print("=" * 65)

    for metric, score in metrics.items():

        if metric == "number_of_images":

            print(f"\nImages evaluated: {score}")

        else:

            print(f"{metric}: {score:.4f}")

    print("\nEvaluation completed successfully!")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
