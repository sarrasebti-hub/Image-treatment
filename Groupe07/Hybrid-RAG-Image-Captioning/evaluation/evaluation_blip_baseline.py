
import json
import torch

from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_IMAGES_DIR = BASE_DIR / "dataset" / "images_val"

HYBRID_PREDICTIONS_PATH = (
    BASE_DIR / "evaluation" / "results" / "predictions.json"
)

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

BLIP_PREDICTIONS_PATH = (
    RESULTS_DIR / "blip_baseline_predictions.json"
)

BLIP_METRICS_PATH = (
    RESULTS_DIR / "blip_baseline_metrics.json"
)

MODEL_NAME = "Salesforce/blip-image-captioning-base"

DEVICE = "cpu"

MAX_NEW_TOKENS = 40
NUM_BEAMS = 3


# ============================================================
# LOAD EXISTING HYBRID PREDICTIONS
# ============================================================

def load_test_images():

    print("\nLoading Hybrid RAG predictions...")

    with open(
        HYBRID_PREDICTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        predictions = json.load(file)

    valid_images = {}

    for image_id, item in predictions.items():

        caption = item.get("generated_caption", "").strip()

        # Exclude the same invalid caption as Hybrid evaluation
        if caption.lower() == (
            "use retrieved captions only as supporting context."
        ):
            print(f"Skipping invalid image: {image_id}")
            continue

        valid_images[str(image_id)] = item

    print(f"\nImages selected: {len(valid_images)}")

    return valid_images


# ============================================================
# LOAD BLIP MODEL
# ============================================================

def load_blip():

    print("\n" + "=" * 60)
    print("LOADING BLIP BASELINE MODEL")
    print("=" * 60)

    processor = BlipProcessor.from_pretrained(
        MODEL_NAME
    )

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    model.eval()

    print("\nBLIP loaded successfully!")

    return processor, model


# ============================================================
# GENERATE BLIP CAPTION
# ============================================================

def generate_caption(image_path, processor, model):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption.strip()


# ============================================================
# GENERATE BASELINE PREDICTIONS
# ============================================================

def generate_all_predictions(
    valid_images,
    processor,
    model
):

    print("\n" + "=" * 60)
    print("GENERATING BLIP-ONLY CAPTIONS")
    print("=" * 60)

    results = {}

    total = len(valid_images)

    for index, (image_id, item) in enumerate(
        valid_images.items(),
        start=1
    ):

        image_name = item["image_name"]

        image_path = TEST_IMAGES_DIR / image_name

        print(f"\n[{index}/{total}] {image_name}")

        if not image_path.exists():

            print("WARNING: Image not found. Skipping.")

            continue

        try:

            caption = generate_caption(
                image_path,
                processor,
                model
            )

            print(f"BLIP: {caption}")

            results[image_id] = {

                "image_id": item["image_id"],

                "image_name": image_name,

                "generated_caption": caption,

                "references": item["references"]

            }

        except Exception as error:

            print(f"ERROR: {error}")

    return results


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(predictions):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        BLIP_PREDICTIONS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            predictions,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nBLIP predictions saved:")
    print(BLIP_PREDICTIONS_PATH)


# ============================================================
# PREPARE COCO DATA
# ============================================================

def prepare_coco_data(predictions):

    references = {}
    generated = {}

    for image_id, item in predictions.items():

        key = int(image_id)

        references[key] = [
            {"caption": caption}
            for caption in item["references"]
        ]

        generated[key] = [
            {"caption": item["generated_caption"]}
        ]

    return references, generated


# ============================================================
# CALCULATE ROUGE-L
# ============================================================

def calculate_rouge_l(predictions):

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

    return sum(scores) / len(scores) if scores else 0.0


# ============================================================
# CALCULATE BLEU AND CIDEr
# ============================================================

def calculate_coco_metrics(references, generated):

    print("\nTokenizing captions...")

    tokenizer = PTBTokenizer()

    tokenized_references = tokenizer.tokenize(
        references
    )

    tokenized_generated = tokenizer.tokenize(
        generated
    )

    print("\nCalculating BLEU...")

    bleu_scorer = Bleu(4)

    bleu_scores, _ = bleu_scorer.compute_score(
        tokenized_references,
        tokenized_generated
    )

    print("BLEU calculated!")

    print("\nCalculating CIDEr...")

    cider_scorer = Cider()

    cider_score, _ = cider_scorer.compute_score(
        tokenized_references,
        tokenized_generated
    )

    print("CIDEr calculated!")

    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-4": float(bleu_scores[3]),
        "CIDEr": float(cider_score)
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("BLIP BASELINE EVALUATION")
    print("=" * 65)

    # Step 1: Select exactly the same valid images
    valid_images = load_test_images()

    if not valid_images:

        print("No valid images found!")

        return

    # Step 2: Load BLIP only
    processor, model = load_blip()

    # Step 3: Generate BLIP predictions
    predictions = generate_all_predictions(
        valid_images,
        processor,
        model
    )

    if not predictions:

        print("No captions generated!")

        return

    # Step 4: Save predictions
    save_predictions(predictions)

    # Step 5: Prepare references and predictions
    references, generated = prepare_coco_data(
        predictions
    )

    # Step 6: Calculate metrics
    metrics = calculate_coco_metrics(
        references,
        generated
    )

    metrics["ROUGE-L"] = calculate_rouge_l(
        predictions
    )

    metrics["Images evaluated"] = len(predictions)

    # Step 7: Save metrics
    with open(
        BLIP_METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Step 8: Display results
    print("\n" + "=" * 65)
    print("FINAL BLIP BASELINE RESULTS")
    print("=" * 65)

    for name, value in metrics.items():

        if isinstance(value, float):

            print(f"{name}: {value:.4f}")

        else:

            print(f"{name}: {value}")

    print("\nMetrics saved:")
    print(BLIP_METRICS_PATH)

    print("\nBLIP BASELINE EVALUATION COMPLETED!")


if __name__ == "__main__":

    main()
