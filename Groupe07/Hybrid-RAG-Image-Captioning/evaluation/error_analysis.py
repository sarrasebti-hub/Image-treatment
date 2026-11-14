
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

HYBRID_PATH = RESULTS_DIR / "predictions.json"
BLIP_PATH = RESULTS_DIR / "blip_baseline_predictions.json"

OUTPUT_PATH = RESULTS_DIR / "error_analysis_10_images.json"


def load_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    hybrid = load_json(HYBRID_PATH)
    blip = load_json(BLIP_PATH)

    common_ids = [
        image_id
        for image_id in hybrid
        if image_id in blip
    ]

    # Select 10 images from the same evaluation set
    selected_ids = common_ids[:10]

    analysis = []

    for image_id in selected_ids:

        hybrid_item = hybrid[image_id]
        blip_item = blip[image_id]

        item = {
            "image_id": image_id,
            "image_name": hybrid_item["image_name"],
            "hybrid_caption": hybrid_item["generated_caption"],
            "blip_caption": blip_item["generated_caption"],
            "references": hybrid_item["references"]
        }

        analysis.append(item)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            analysis,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print("ERROR ANALYSIS - 10 IMAGES")
    print("=" * 60)

    for item in analysis:

        print(f"\nImage: {item['image_name']}")

        print(f"BLIP: {item['blip_caption']}")

        print(f"Hybrid RAG: {item['hybrid_caption']}")

        print("\nGround Truth:")

        for reference in item["references"]:
            print("-", reference)

        print("-" * 60)

    print("\nAnalysis saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
