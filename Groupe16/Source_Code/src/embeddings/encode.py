import random
import os

import numpy as np
from tqdm import tqdm
from time import perf_counter

from src.preprocessing.metadata import load_metadata, load_rgb
from src.utils.core import digest, file_hash, normalize_vectors, path, read_json, stamp, write_json


def check_matrix(vectors, rows, dimension=None):
    if vectors.ndim != 2 or vectors.shape[0] != len(rows) or not len(rows):
        raise ValueError("Embedding row count/dimensions mismatch")
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError("Embedding feature dimension mismatch")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite float32 (no NaN/Inf)")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Embedding vectors are not L2-normalized")
    keys = [(row["image_id"], row.get("caption_id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate embedding row IDs")


def input_identity(cfg, records):
    """Fingerprint only the actual image/text encoder inputs, not analysis metadata."""
    return digest({"identity_schema_version": 2, "embedding": cfg["embedding"], "seed": cfg["random_seed"],
                   "preprocessing": cfg["preprocessing"], "records": encoder_inputs(records)})


def encoder_inputs(records):
    return [{"image_id": r["image_id"], "file_sha256": r["file_sha256"],
             "coco_source": r["coco_source"], "file_name": r["file_name"],
             "captions": [{"caption_id": c["caption_id"], "caption_clean": c["caption_clean"]} for c in r["captions"]]}
            for r in records]


def legacy_input_identity(cfg, records):
    return digest({"embedding": cfg["embedding"], "seed": cfg["random_seed"],
                   "preprocessing": cfg["preprocessing"], "records": records})


def load_embeddings(cfg, records=None):
    if records is None:
        records = load_metadata(cfg, "knowledge_base")
    folder = path(cfg, "embeddings")
    manifest = read_json(folder / "manifest.json")
    identity = input_identity if manifest.get("identity_schema_version") == 2 else legacy_input_identity
    if manifest["input_sha256"] != identity(cfg, records):
        raise ValueError("Embeddings stale: inputs/configuration differ; rerun embeddings --force")
    arrays, rows = {}, {}
    for kind in ("image", "text"):
        if any(file_hash(folder / name) != sha for name, sha in manifest[kind]["files_sha256"].items()):
            raise ValueError(f"{kind}: embedding artifacts changed")
        arrays[kind] = np.load(folder / f"{kind}.npy", allow_pickle=False)
        rows[kind] = read_json(folder / f"{kind}_rows.json")
        check_matrix(arrays[kind], rows[kind], manifest[kind]["dimension"])
    expected_image = [{"image_id": r["image_id"]} for r in records]
    expected_text = [{"image_id": r["image_id"], "caption_id": c["caption_id"]} for r in records for c in r["captions"]]
    if rows["image"] != expected_image or rows["text"] != expected_text:
        raise ValueError("ID-to-embedding row association mismatch")
    return arrays, rows, manifest


class Encoder:
    def __init__(self, cfg):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        self.torch = torch
        self.cfg = cfg
        torch.set_num_threads(cfg["embedding"].get("cpu_threads", 8))
        random.seed(cfg["random_seed"])
        np.random.seed(cfg["random_seed"])
        torch.manual_seed(cfg["random_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["random_seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        requested = cfg["embedding"]["device"]
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
        cache = str(path(cfg, "cache") / "huggingface")
        settings = dict(revision=cfg["embedding"]["revision"], cache_dir=cache, trust_remote_code=False,
                        local_files_only=cfg.get("_offline", False))
        # Resolve before loading: newer Transformers versions may omit config._commit_hash.
        config_file = hf_hub_download(repo_id=cfg["embedding"]["model"], filename="config.json",
                                      revision=settings["revision"], cache_dir=cache,
                                      local_files_only=settings["local_files_only"])
        from pathlib import Path
        self.resolved_revision = Path(config_file).parent.name
        settings["revision"] = self.resolved_revision
        self.model = AutoModel.from_pretrained(cfg["embedding"]["model"], **settings).to(self.device).eval()
        # Resolve processor to the exact same immutable revision as the weights.
        if self.resolved_revision:
            settings["revision"] = self.resolved_revision
        self.shared_space = hasattr(self.model, "get_image_features") and hasattr(self.model, "get_text_features")
        self.image_only = self.model.config.model_type == "dinov2"
        if not self.shared_space and not self.image_only:
            raise ValueError("Supported encoders: CLIP/SigLIP dual encoders, or DINOv2 with separate text model")
        if self.image_only:
            self.processor = AutoImageProcessor.from_pretrained(cfg["embedding"]["model"], **settings)
            text_settings = dict(revision=cfg["embedding"]["fallback_text_revision"], cache_dir=cache, trust_remote_code=False)
            self.text_model = AutoModel.from_pretrained(cfg["embedding"]["fallback_text_model"], **text_settings).to(self.device).eval()
            self.text_revision = getattr(self.text_model.config, "_commit_hash", None)
            if self.text_revision:
                text_settings["revision"] = self.text_revision
            self.text_processor = AutoProcessor.from_pretrained(cfg["embedding"]["fallback_text_model"], **text_settings)
        else:
            self.processor = AutoProcessor.from_pretrained(cfg["embedding"]["model"], **settings)
            self.text_model, self.text_processor = self.model, self.processor
            self.text_revision = self.resolved_revision
        self.truncated_captions = 0

    def features(self, output):
        # Transformers APIs can return a tensor or BaseModelOutputWithPooling.
        if hasattr(output, "pooler_output"):
            output = output.pooler_output
        if not self.torch.is_tensor(output):
            raise TypeError("Unsupported encoder output; expected projected feature tensor")
        return normalize_vectors(output.detach().float().cpu().numpy())

    def images(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            if self.image_only:
                return self.features(self.model(**inputs).last_hidden_state[:, 0, :])
            return self.features(self.model.get_image_features(**inputs))

    def texts(self, captions):
        maximum = self.text_model.config.text_config.max_position_embeddings
        lengths = self.text_processor.tokenizer(captions, truncation=False)["input_ids"]
        self.truncated_captions += sum(len(tokens) > maximum for tokens in lengths)
        padding = self.cfg["embedding"].get("text_padding", "max_length")
        inputs = self.text_processor(text=captions, padding=padding, truncation=True, max_length=maximum, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            return self.features(self.text_model.get_text_features(**inputs))


def generate_embeddings(cfg, data=None, force=False):
    if data is None:
        from src.data.coco import CocoFiles
        data = CocoFiles(cfg)
    records = load_metadata(cfg, "knowledge_base")
    folder = path(cfg, "embeddings")
    if (folder / "manifest.json").exists() and not force:
        _, _, manifest = load_embeddings(cfg, records)
        for row in records:
            filename = data.image_path(row["coco_source"], row)
            if file_hash(filename) != row["file_sha256"]:
                raise ValueError("Image changed since metadata generation")
        return {"status": "reused_verified", **manifest}
    encoder = Encoder(cfg)
    started = perf_counter()
    batch_size = cfg["embedding"]["batch_size"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    identity = input_identity(cfg, records)
    checkpoints = folder / "checkpoints" / identity[:16]
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_file = folder / "progress.json"
    completed = {"image": 0, "text": 0}

    def checkpoint(kind, start, batch_rows, compute):
        filename = checkpoints / f"{kind}_{start:06d}.npy"
        if filename.exists() and not force:
            values = np.load(filename, allow_pickle=False)
            check_matrix(values, batch_rows)
        else:
            values = compute()
            check_matrix(values, batch_rows)
            temporary = filename.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(filename)
        completed[kind] += len(batch_rows)
        write_json(progress_file, {"input_sha256": identity, "completed": completed,
                                   "expected": {"image": len(records), "text": sum(len(r["captions"]) for r in records)},
                                   "elapsed_seconds_this_run": perf_counter() - started})
        return values

    image_batches = []
    for start in tqdm(range(0, len(records), batch_size), desc="Image embeddings"):
        batch = records[start:start + batch_size]
        images = []
        cached = (checkpoints / f"image_{start:06d}.npy").exists() and not force
        try:
            for row in batch:
                filename = data.image_path(row["coco_source"], row)
                if file_hash(filename) != row["file_sha256"]:
                    raise ValueError(f"Image {row['image_id']} changed since metadata creation")
                if not cached:
                    images.append(load_rgb(filename))
            image_batches.append(checkpoint("image", start, [{"image_id": r["image_id"]} for r in batch], lambda: encoder.images(images)))
        finally:
            for image in images:
                image.close()
    captions = [c["caption_clean"] for r in records for c in r["captions"]]
    text_rows = [{"image_id": r["image_id"], "caption_id": c["caption_id"]} for r in records for c in r["captions"]]
    text_batches = [checkpoint("text", start, text_rows[start:start + batch_size], lambda start=start: encoder.texts(captions[start:start + batch_size]))
                    for start in tqdm(range(0, len(captions), batch_size), desc="Text embeddings", mininterval=5)]
    arrays = {"image": np.concatenate(image_batches), "text": np.concatenate(text_batches)}
    rows = {"image": [{"image_id": r["image_id"]} for r in records],
            "text": text_rows}
    shared_space = getattr(encoder, "shared_space", True)
    if shared_space and arrays["image"].shape[1] != arrays["text"].shape[1]:
        raise ValueError("Image/text embeddings must share a multimodal space")
    manifest = {**stamp(cfg), "input_sha256": input_identity(cfg, records),
                "identity_schema_version": 2,
                "model": cfg["embedding"]["model"], "requested_revision": cfg["embedding"]["revision"],
                "resolved_revision": encoder.resolved_revision, "device": encoder.device,
                "shared_image_text_space": shared_space,
                "text_model": cfg["embedding"]["model"] if shared_space else cfg["embedding"]["fallback_text_model"],
                "text_resolved_revision": getattr(encoder, "text_revision", encoder.resolved_revision),
                "batch_size": batch_size, "dtype": "float32", "normalization": "L2 per row",
                "text_truncation_count": encoder.truncated_captions, "query_policy": "Only KB reference captions encoded; validation/test references excluded"}
    # Count truncations independently of checkpoints so resumed runs have correct totals.
    if hasattr(encoder, "text_processor"):
        maximum = encoder.text_model.config.text_config.max_position_embeddings
        tokens = encoder.text_processor.tokenizer(captions, truncation=False)["input_ids"]
        manifest["text_truncation_count"] = sum(len(t) > maximum for t in tokens)
    manifest["elapsed_seconds_this_run"] = perf_counter() - started
    folder.mkdir(parents=True, exist_ok=True)
    for kind in arrays:
        check_matrix(arrays[kind], rows[kind])
        np.save(folder / f"{kind}.npy", arrays[kind], allow_pickle=False)
        write_json(folder / f"{kind}_rows.json", rows[kind])
        manifest[kind] = {"dimension": arrays[kind].shape[1], "vectors": len(rows[kind]),
                          "files_sha256": {name: file_hash(folder / name) for name in (f"{kind}.npy", f"{kind}_rows.json")}}
    write_json(folder / "manifest.json", manifest)
    return manifest
