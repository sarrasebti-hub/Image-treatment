
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def generate_sbert_caption_embeddings(
    caption_records,
    output_dir,
    batch_size=64,
    model_name=SBERT_MODEL_NAME,
    device="cpu",
    show_progress=True,
):
    """
    Generate L2-normalized Sentence-BERT embeddings for captions.

    Parameters
    ----------
    caption_records : list[dict]
        COCO caption annotation records. Each record must contain:
        id, image_id, caption.

    output_dir : str or Path
        Directory where embeddings and mapping will be saved.

    batch_size : int
        Number of captions processed per batch.

    model_name : str
        Sentence-Transformer model identifier.

    device : str
        Computation device, for example 'cpu' or 'cuda'.

    show_progress : bool
        Whether to display the encoding progress bar.

    Returns
    -------
    dict
        Summary of the embedding generation process.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check input records
    if not caption_records:
        raise ValueError("No caption records were provided.")

    required_fields = {"id", "image_id", "caption"}

    for record in caption_records:
        missing_fields = required_fields - set(record.keys())

        if missing_fields:
            raise ValueError(
                f"Caption record is missing fields: {missing_fields}"
            )

        if not isinstance(record["caption"], str):
            raise TypeError("Every caption must be a string.")

        if not record["caption"].strip():
            raise ValueError("Empty caption encountered.")

    # Sort captions by their unique COCO annotation ID
    caption_records = sorted(
        caption_records,
        key=lambda record: record["id"]
    )

    caption_ids = [record["id"] for record in caption_records]

    if len(set(caption_ids)) != len(caption_ids):
        raise ValueError("Duplicate caption IDs detected.")

    captions = [
        record["caption"].strip()
        for record in caption_records
    ]

    print(f"Loading Sentence-BERT model: {model_name}")
    print(f"Device: {device}")

    model = SentenceTransformer(
        model_name,
        device=device
    )

    embedding_dimension = model.get_sentence_embedding_dimension()

    # Generate embeddings with L2 normalization
    embeddings = model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Validate generated embeddings
    expected_shape = (
        len(caption_records),
        embedding_dimension
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embedding values detected.")

    # Build row-to-caption mapping
    caption_mapping = []

    for row, record in enumerate(caption_records):
        caption_mapping.append({
            "row": row,
            "caption_id": record["id"],
            "image_id": record["image_id"],
        })

    # Save embedding matrix
    embeddings_file = output_dir / "sbert_caption_embeddings.npy"

    np.save(embeddings_file, embeddings)

    # Save mapping
    mapping_file = output_dir / "caption_embedding_index.json"

    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(
            caption_mapping,
            f,
            ensure_ascii=False,
            indent=2
        )

    # Calculate embedding norms for metadata
    embedding_norms = np.linalg.norm(
        embeddings,
        ord=2,
        axis=1
    )

    summary = {
        "model_name": model_name,
        "number_of_captions": len(captions),
        "embedding_dimension": embedding_dimension,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "normalized": True,
        "normalization_method": "L2",
        "device": device,
        "batch_size": batch_size,
        "minimum_l2_norm": float(embedding_norms.min()),
        "maximum_l2_norm": float(embedding_norms.max()),
        "embeddings_file": str(embeddings_file),
        "mapping_file": str(mapping_file),
        "status": "success",
    }

    print("\nSentence-BERT embedding generation completed.")
    print("Shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print("Embeddings saved:", embeddings_file)
    print("Mapping saved:", mapping_file)

    return summary
