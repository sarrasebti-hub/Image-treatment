
import sys
import argparse
import re
import torch
import open_clip

from pathlib import Path
from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEVICE = "cpu"

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
FLAN_MODEL_NAME = "google/flan-t5-small"
YOLO_MODEL_NAME = "yolo26n.pt"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


# ============================================================
# IMPORT RETRIEVAL MODULES
# ============================================================

from retrieval.hybrid_retrieval import HybridRetriever
from retrieval.caption_retrieval import CaptionRetriever


# ============================================================
# CAPTION GENERATION SYSTEM
# ============================================================

class CaptionGenerationSystem:

    def __init__(self):

        print("=" * 65)
        print("INITIALIZING HYBRID RAG CAPTION GENERATION")
        print("=" * 65)

        self.device = DEVICE

        self.blip_processor = None
        self.blip_model = None

        self.flan_tokenizer = None
        self.flan_model = None

        self.clip_model = None
        self.clip_preprocess = None

        self.yolo_model = None

        self.hybrid_retriever = HybridRetriever(
            rrf_k=60
        )

        self.caption_retriever = CaptionRetriever()

    # ========================================================
    # LOAD BLIP
    # ========================================================

    def load_blip(self):

        print("\nLoading BLIP model...")

        self.blip_processor = BlipProcessor.from_pretrained(
            BLIP_MODEL_NAME
        )

        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME
        ).to(self.device)

        self.blip_model.eval()

        print("BLIP model loaded successfully!")

    # ========================================================
    # LOAD FLAN-T5
    # ========================================================

    def load_flan(self):

        print("\nLoading FLAN-T5 model...")

        self.flan_tokenizer = AutoTokenizer.from_pretrained(
            FLAN_MODEL_NAME
        )

        self.flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_NAME
        ).to(self.device)

        self.flan_model.eval()

        print("FLAN-T5 model loaded successfully!")

    # ========================================================
    # LOAD CLIP
    # ========================================================

    def load_clip(self):

        print("\nLoading CLIP model...")

        (
            self.clip_model,
            _,
            self.clip_preprocess
        ) = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            force_quick_gelu=True
        )

        self.clip_model = self.clip_model.to(
            self.device
        )

        self.clip_model.eval()

        print("CLIP model loaded successfully!")

    # ========================================================
    # LOAD YOLO
    # ========================================================

    def load_yolo(self):

        print("\nLoading YOLO model...")

        from ultralytics import YOLO

        self.yolo_model = YOLO(
            YOLO_MODEL_NAME
        )

        print("YOLO model loaded successfully!")

    # ========================================================
    # INITIALIZE MODELS AND INDEXES
    # ========================================================

    def initialize(self):

        self.load_blip()
        self.load_flan()
        self.load_clip()
        self.load_yolo()

        print("\nInitializing retrieval indexes...")

        self.hybrid_retriever.build_indexes()

        self.caption_retriever.build_index()

        print("\nAll models and indexes are ready!")

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def load_image(self, image_path):

        image_path = Path(image_path)

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        return Image.open(image_path).convert("RGB")

    # ========================================================
    # BLIP INITIAL CAPTION
    # ========================================================

    def generate_initial_caption(self, image):

        print("\nGenerating initial caption with BLIP...")

        inputs = self.blip_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            output = self.blip_model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=3
            )

        caption = self.blip_processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption.strip()

    # ========================================================
    # CLIP IMAGE EMBEDDING
    # ========================================================

    def extract_clip_embedding(self, image):

        print("\nExtracting CLIP image embedding...")

        image_tensor = self.clip_preprocess(
            image
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():

            embedding = self.clip_model.encode_image(
                image_tensor
            )

            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True
            )

        return embedding.cpu().numpy()

    # ========================================================
    # YOLO OBJECT DETECTION
    # ========================================================

    def extract_objects(self, image_path):

        print("\nExtracting objects with YOLO...")

        results = self.yolo_model.predict(
            source=str(image_path),
            conf=0.25,
            verbose=False
        )

        detected_labels = []

        for result in results:

            if result.boxes is None:
                continue

            for class_id in result.boxes.cls.tolist():

                label = result.names[int(class_id)]

                detected_labels.append(label)

        detected_labels = list(
            dict.fromkeys(detected_labels)
        )

        print("Detected objects:", detected_labels)

        return detected_labels

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def retrieve_similar_images(
        self,
        image_embedding,
        detected_labels,
        initial_caption
    ):

        print("\nRunning Hybrid Retrieval...")

        query_text = " ".join(detected_labels)

        if not query_text.strip():
            query_text = initial_caption

        results = self.hybrid_retriever.search(
            query_embedding=image_embedding,
            query_text=query_text,
            top_k=5,
            candidate_k=50
        )

        print("\nTop Hybrid Retrieval Results:")

        for rank, result in enumerate(results, start=1):

            print(
                f"{rank}. {result['image_name']} "
                f"| RRF: {result['rrf_score']:.6f}"
            )

        return results

    # ========================================================
    # CAPTION RETRIEVAL
    # ========================================================

    def retrieve_captions(self, hybrid_results):

        print("\nRetrieving related captions...")

        results = self.caption_retriever.search(
            hybrid_results
        )

        total = 0

        for result in results:

            captions = result.get("captions", [])

            total += len(captions)

            print(
                f"\nImage: {result['image_name']}"
            )

            for caption in captions:
                print("-", caption)

        print(f"\nTotal retrieved captions: {total}")

        return results

    # ========================================================
    # KEYWORD EXTRACTION
    # ========================================================

    @staticmethod
    def extract_keywords(text):

        stop_words = {
            "a", "an", "the", "is", "are", "in", "on",
            "of", "with", "and", "to", "for", "by",
            "at", "from", "this", "that", "there",
            "it", "its", "under", "over", "along"
        }

        words = re.findall(
            r"\b[a-z]+\b",
            text.lower()
        )

        return {
            word for word in words
            if word not in stop_words and len(word) > 2
        }

    # ========================================================
    # BUILD RAG CONTEXT
    # ========================================================

    def build_context(
        self,
        initial_caption,
        detected_labels,
        retrieved_captions
    ):

        print("\nBuilding RAG context...")

        reference_keywords = self.extract_keywords(
            initial_caption
        )

        for label in detected_labels:

            reference_keywords.update(
                self.extract_keywords(label)
            )

        candidates = []

        for result in retrieved_captions:

            for caption in result.get("captions", []):

                keywords = self.extract_keywords(caption)

                common_words = (
                    reference_keywords & keywords
                )

                if len(common_words) >= 2:

                    candidates.append({
                        "caption": caption,
                        "score": len(common_words)
                    })

        # Remove duplicate captions
        unique_candidates = {}

        for item in candidates:

            caption = item["caption"].strip()

            if caption not in unique_candidates:

                unique_candidates[caption] = item

        candidates = list(unique_candidates.values())

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        selected_captions = [
            item["caption"]
            for item in candidates[:3]
        ]

        if selected_captions:

            retrieved_context = "\n".join(
                selected_captions
            )

        else:

            retrieved_context = (
                "No sufficiently relevant captions found."
            )

        object_text = (
            ", ".join(detected_labels)
            if detected_labels
            else "No objects detected."
        )

        prompt = f"""
Write one concise English caption describing an image.

Initial image description:
{initial_caption}

Detected objects:
{object_text}

Descriptions of similar images:
{retrieved_context}

Instructions:
- Preserve the main meaning of the initial description.
- Use retrieved descriptions only as supplementary context.
- Do not copy details from different images.
- Do not invent objects, people, actions or locations.
- Write one natural and grammatically correct sentence.
- Return only the caption.

Caption:
""".strip()

        print("\nSelected retrieved captions:")

        for caption in selected_captions:
            print("-", caption)

        print("\nGenerated RAG Prompt:")
        print(prompt)

        return prompt

    # ========================================================
    # VALIDATE CAPTION
    # ========================================================

    def is_valid_caption(self, caption):

        if not caption or not isinstance(caption, str):
            return False

        caption = caption.strip()

        lower_caption = caption.lower()

        invalid_phrases = [
            "objects detected",
            "related captions",
            "write only",
            "generate one",
            "visual description",
            "instructions",
            "blip description",
            "detected objects",
            "retrieved descriptions",
            "supporting information",
            "never copy",
            "do not invent",
            "do not mention",
            "caption assistant",
            "describe the input image",
            "additional descriptions",
            "use detected objects",
            "return only",
            "english sentence",
            "output only the caption"
        ]

        if any(
            phrase in lower_caption
            for phrase in invalid_phrases
        ):
            return False

        words = caption.split()

        if len(words) < 4 or len(words) > 30:
            return False

        if "\n" in caption:
            return False

        if "|" in caption:
            return False

        if " - " in caption:
            return False

        if caption.count(".") > 1:
            return False

        if caption.endswith(":"):
            return False

        return True

    # ========================================================
    # FLAN-T5 GENERATION
    # ========================================================

    def generate_final_caption(
        self,
        prompt,
        fallback_caption
    ):

        print("\nGenerating final caption with FLAN-T5...")

        inputs = self.flan_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            output = self.flan_model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True
            )

        caption = self.flan_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        print("\nRaw FLAN-T5 output:", repr(caption))
        print("Word count:", len(caption.split()))

        valid = self.is_valid_caption(caption)

        print("Validation result:", valid)

        if valid:

            print("\nCaption validation: PASSED")

            return caption, False

        print("\nCaption validation: FAILED")
        print("Using BLIP fallback.")

        return fallback_caption.strip(), True

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def generate(self, image_path):

        print("\n" + "=" * 65)
        print("STARTING HYBRID RAG CAPTION GENERATION")
        print("=" * 65)

        image_path = Path(image_path)

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        image = self.load_image(image_path)

        print("\nInput image:", image_path)

        # STEP 1: BLIP
        initial_caption = self.generate_initial_caption(image)

        print("\nBLIP caption:", initial_caption)

        # STEP 2: CLIP
        image_embedding = self.extract_clip_embedding(image)

        # STEP 3: YOLO
        detected_labels = self.extract_objects(image_path)

        # STEP 4: HYBRID RETRIEVAL
        hybrid_results = self.retrieve_similar_images(
            image_embedding,
            detected_labels,
            initial_caption
        )

        # STEP 5: CAPTION RETRIEVAL
        retrieved_captions = self.retrieve_captions(
            hybrid_results
        )

        # STEP 6: RAG CONTEXT
        prompt = self.build_context(
            initial_caption,
            detected_labels,
            retrieved_captions
        )

        # STEP 7: FLAN-T5
        final_caption, used_fallback = (
            self.generate_final_caption(
                prompt,
                initial_caption
            )
        )

        print("\n" + "=" * 65)
        print("FINAL GENERATED CAPTION")
        print("=" * 65)

        print("\n", final_caption)

        print("\nUsed BLIP fallback:", used_fallback)

        print("\n" + "=" * 65)
        print("CAPTION GENERATION COMPLETED")
        print("=" * 65)

        return {
            "image_path": str(image_path),
            "initial_caption": initial_caption,
            "detected_objects": detected_labels,
            "retrieved_images": hybrid_results,
            "retrieved_captions": retrieved_captions,
            "final_caption": final_caption,
            "used_fallback": used_fallback
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Hybrid RAG Image Captioning"
    )

    parser.add_argument(
        "image",
        help="Path to the input image"
    )

    args = parser.parse_args()

    try:

        system = CaptionGenerationSystem()

        system.initialize()

        system.generate(args.image)

    except Exception as error:

        print("\nERROR:", error)

        raise
