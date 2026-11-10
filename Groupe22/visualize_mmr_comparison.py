import json
from pathlib import Path
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    BASE_DIR / "dataset" / "embeddings"
    / "mmr_parameter_results.json"
)

IMAGE_DIR = BASE_DIR / "dataset" / "dataset" / "train2017"

OUTPUT_DIR = BASE_DIR / "dataset" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDAS = ["0.5", "0.7", "0.9"]

TILE_WIDTH = 300
TILE_HEIGHT = 200
IMAGE_HEIGHT = 135

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Select the first 3 validation queries
selected_queries = data[:3]

for query in selected_queries:

    query_id = query["query_id"]

    sheet = Image.new(
        "RGB",
        (TILE_WIDTH * 3, TILE_HEIGHT * 5),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for col, lam in enumerate(LAMBDAS):

        results = query["lambda_results"][lam]

        for row, result in enumerate(results):

            image_id = result["image_id"]
            caption = result["caption"]
            score = result["relevance_score"]

            image_path = IMAGE_DIR / f"{image_id}.jpg"

            x = col * TILE_WIDTH
            y = row * TILE_HEIGHT

            if image_path.exists():

                img = Image.open(image_path).convert("RGB")
                img.thumbnail((TILE_WIDTH, IMAGE_HEIGHT))

                sheet.paste(img, (x, y + 30))

            else:
                draw.text(
                    (x + 10, y + 50),
                    f"Image not found: {image_id}",
                    fill="red"
                )

            draw.text(
                (x + 5, y + 5),
                f"Lambda = {lam}",
                fill="blue"
            )

            draw.text(
                (x + 5, y + IMAGE_HEIGHT + 35),
                f"ID: {image_id} | Score: {score:.3f}",
                fill="black"
            )

            draw.text(
                (x + 5, y + IMAGE_HEIGHT + 55),
                caption[:65],
                fill="black"
            )

    output_file = (
        OUTPUT_DIR / f"mmr_visual_comparison_{query_id}.jpg"
    )

    sheet.save(output_file)

    print("Saved:", output_file)

print("\nVisual comparison completed.")