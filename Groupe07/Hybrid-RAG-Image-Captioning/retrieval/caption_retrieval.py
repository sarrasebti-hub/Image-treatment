
import json

from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

CAPTIONS_PATH = (
    BASE_DIR / "dataset" / "processed" / "train_captions.json"
)


class CaptionRetriever:

    def __init__(self):

        self.captions_by_image = defaultdict(list)
        self.image_metadata = {}

    def build_index(self):

        print("=" * 60)
        print("BUILDING CAPTION RETRIEVAL INDEX")
        print("=" * 60)

        print("\nLoading COCO captions...")

        if not CAPTIONS_PATH.exists():
            raise FileNotFoundError(
                f"Captions file not found: {CAPTIONS_PATH}"
            )

        with open(
            CAPTIONS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # Store image metadata
        for image in data["images"]:

            image_id = int(image["id"])

            self.image_metadata[image_id] = {
                "file_name": image["file_name"],
                "width": image.get("width"),
                "height": image.get("height")
            }

        # Group captions by image_id
        for annotation in data["annotations"]:

            image_id = int(annotation["image_id"])

            caption = annotation["caption"].strip()

            if caption:
                self.captions_by_image[image_id].append(caption)

        print("\nCaption index successfully created!")

        print(f"Images indexed: {len(self.image_metadata)}")

        print(
            f"Captions indexed: "
            f"{sum(len(c) for c in self.captions_by_image.values())}"
        )

        print(
            f"Images with captions: "
            f"{len(self.captions_by_image)}"
        )

    def search(self, hybrid_results):

        if not self.captions_by_image:
            raise RuntimeError(
                "Build the caption index first."
            )

        results = []

        for result in hybrid_results:

            image_id = int(result["image_id"])

            captions = self.captions_by_image.get(
                image_id,
                []
            )

            metadata = self.image_metadata.get(
                image_id,
                {}
            )

            results.append({

                "image_id": image_id,

                "image_name": result["image_name"],

                "rrf_score": result["rrf_score"],

                "captions": captions,

                "number_of_captions": len(captions),

                "image_metadata": metadata

            })

        return results


if __name__ == "__main__":

    retriever = CaptionRetriever()

    # Build caption index
    retriever.build_index()

    # Example: test with known image IDs
    test_results = [

        {
            "image_id": 419074,
            "image_name": "000000419074.jpg",
            "rrf_score": 0.032
        },

        {
            "image_id": 352185,
            "image_name": "000000352185.jpg",
            "rrf_score": 0.028
        }

    ]

    results = retriever.search(test_results)

    print("\n" + "=" * 60)
    print("TESTING CAPTION RETRIEVAL")
    print("=" * 60)

    for result in results:

        print(f"\nImage: {result['image_name']}")

        print(f"Image ID: {result['image_id']}")

        print(f"RRF Score: {result['rrf_score']:.6f}")

        print(
            f"Number of captions: "
            f"{result['number_of_captions']}"
        )

        print("\nRetrieved Captions:")

        for index, caption in enumerate(
            result["captions"],
            start=1
        ):

            print(f"{index}. {caption}")
