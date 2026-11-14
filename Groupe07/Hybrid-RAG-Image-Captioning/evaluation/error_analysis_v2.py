
import json
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

BLIP_PATH = RESULTS_DIR / "blip_baseline_predictions.json"
V1_PATH = RESULTS_DIR / "predictions.json"
V2_PATH = RESULTS_DIR / "hybrid_rag_v2_predictions.json"

OUTPUT_PATH = RESULTS_DIR / "error_analysis_v2.json"

SUSPICIOUS_PHRASES = [
    "detected objects",
    "similar image",
    "retrieved captions",
    "write one concise",
    "return one complete sentence",
    "supporting context",
    "do not copy",
    "describe only",
    "initial caption",
    "generate one caption",
    "output only",
    "caption assistant"
]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize(text):
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_caption(item):
    return item.get("generated_caption", "").strip()


def detect_prompt_leakage(caption):
    normalized = normalize(caption)

    return any(
        phrase in normalized
        for phrase in SUSPICIOUS_PHRASES
    )


def detect_incomplete_caption(caption):
    words = caption.strip().split()

    if len(words) < 4:
        return True

    if caption.endswith(("...", "-", "|", ":")):
        return True

    return False


def detect_possible_contamination(caption):
    normalized = normalize(caption)

    return (
        "|" in caption
        or "\n" in caption
        or "similar image has" in normalized
        or "detected objects:" in normalized
    )


def compare_with_blip(caption, blip_caption):
    return normalize(caption) == normalize(blip_caption)


def analyze_item(image_id, blip, v1, v2):

    blip_item = blip[image_id]
    v1_item = v1[image_id]
    v2_item = v2[image_id]

    blip_caption = get_caption(blip_item)
    v1_caption = get_caption(v1_item)
    v2_caption = get_caption(v2_item)

    fallback = v2_item.get("used_fallback", False)

    issues = []

    if detect_prompt_leakage(v2_caption):
        issues.append("Possible prompt leakage")

    if detect_incomplete_caption(v2_caption):
        issues.append("Possibly incomplete caption")

    if detect_possible_contamination(v2_caption):
        issues.append("Possible caption contamination")

    if compare_with_blip(v2_caption, blip_caption):
        issues.append("V2 caption identical to BLIP")

    if fallback:
        issues.append("BLIP fallback used")

    return {
        "image_id": image_id,
        "image_name": v2_item.get(
            "image_name",
            v1_item.get("image_name", "")
        ),
        "blip_caption": blip_caption,
        "hybrid_v1_caption": v1_caption,
        "hybrid_v2_caption": v2_caption,
        "initial_caption": v2_item.get("initial_caption", ""),
        "used_fallback": fallback,
        "references": v2_item.get(
            "references",
            v1_item.get("references", [])
        ),
        "analysis": {
            "prompt_leakage": detect_prompt_leakage(v2_caption),
            "incomplete_caption": detect_incomplete_caption(v2_caption),
            "possible_contamination": detect_possible_contamination(
                v2_caption
            ),
            "identical_to_blip": compare_with_blip(
                v2_caption,
                blip_caption
            ),
            "fallback_used": fallback,
            "possible_issues": issues
        }
    }


def main():

    print("=" * 65)
    print("HYBRID RAG V2 - ERROR ANALYSIS")
    print("=" * 65)

    blip = load_json(BLIP_PATH)
    v1 = load_json(V1_PATH)
    v2 = load_json(V2_PATH)

    common_ids = [
        image_id
        for image_id in v2
        if image_id in v1 and image_id in blip
    ]

    if not common_ids:
        raise ValueError(
            "No common image IDs found between BLIP, V1 and V2."
        )

    results = []

    for image_id in common_ids:
        result = analyze_item(image_id, blip, v1, v2)
        results.append(result)

    # Calculate statistics
    total = len(results)

    fallback_count = sum(
        item["analysis"]["fallback_used"]
        for item in results
    )

    identical_count = sum(
        item["analysis"]["identical_to_blip"]
        for item in results
    )

    leakage_count = sum(
        item["analysis"]["prompt_leakage"]
        for item in results
    )

    incomplete_count = sum(
        item["analysis"]["incomplete_caption"]
        for item in results
    )

    contamination_count = sum(
        item["analysis"]["possible_contamination"]
        for item in results
    )

    summary = {
        "number_of_images": total,
        "fallback_count": fallback_count,
        "fallback_rate_percent": round(
            fallback_count / total * 100, 2
        ),
        "identical_to_blip_count": identical_count,
        "prompt_leakage_count": leakage_count,
        "incomplete_caption_count": incomplete_count,
        "possible_contamination_count": contamination_count
    }

    output = {
        "summary": summary,
        "results": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Console report
    print("\nSUMMARY")
    print("-" * 65)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFIRST 10 IMAGE ANALYSES")
    print("-" * 65)

    for item in results[:10]:

        print(f"\nImage: {item['image_name']}")
        print(f"BLIP: {item['blip_caption']}")
        print(f"V1: {item['hybrid_v1_caption']}")
        print(f"V2: {item['hybrid_v2_caption']}")

        print(
            "Fallback:",
            item["analysis"]["fallback_used"]
        )

        print(
            "Possible issues:",
            item["analysis"]["possible_issues"]
        )

    print("\n" + "=" * 65)
    print("ERROR ANALYSIS V2 COMPLETED")
    print("=" * 65)
    print(f"Total images analyzed: {total}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
