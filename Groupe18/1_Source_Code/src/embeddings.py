"""Resumable CLIP image embeddings for the immutable selected manifests."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _device(config):
    requested = config.get("embedding_device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_rows(root: Path, split: str):
    manifest = json.loads((root / "data" / "selected" / f"{split}_ids.json").read_text(encoding="utf-8"))
    out = []
    for iid in manifest["image_ids"]:
        path = root / "data" / "processed" / split / f"{iid:012d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Processed image missing for {split} image_id={iid}: {path}")
        out.append({"image_id": int(iid), "split": split, "filename": path.name,
                    "processed_path": path.relative_to(root).as_posix()})
    return out


def _revision(model_name: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_name, local_files_only=True)).name
    except Exception:
        return None


def _valid_cache(emb_path: Path, meta_path: Path, rows, model_name: str, dim: int, revision: str | None):
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.load(emb_path, mmap_mode="r")
        prior_rows=meta.get("embeddings",[])
        mapping_ok=len(prior_rows)==len(rows) and all(
            int(old.get("image_id",-1))==int(new["image_id"]) and old.get("processed_path")==new["processed_path"]
            and old.get("split")==new["split"] for old,new in zip(prior_rows,rows))
        return (arr.shape == (len(rows), dim) and np.isfinite(arr).all() and mapping_ok
                and meta.get("model") == model_name and meta.get("count") == len(rows)
                and meta.get("normalization") == "L2" and len(meta.get("embeddings", [])) == len(rows)
                and meta.get("model_revision") == revision)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def extract_all(root: Path, config: dict):
    model_name = config["embedding_models"]["image"]
    model_revision = _revision(model_name)
    out_dir = root / config.get("embedding_output_dir", "embeddings/image_embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {s: _load_rows(root, s) for s in ("train", "validation", "test")}
    # Reuse complete, verified split caches without loading weights.
    try:
        from transformers import CLIPConfig
        model_cfg = CLIPConfig.from_pretrained(model_name, local_files_only=True)
        dim = int(model_cfg.projection_dim)
    except Exception:
        dim = 512  # CLIP ViT-B/32 projection dimension; confirmed against model outputs below.
    names = {}
    for split, rows in splits.items():
        ep, mp = out_dir / f"{split}.npy", out_dir / f"{split}.metadata.json"
        if _valid_cache(ep, mp, rows, model_name, dim, model_revision):
            names[split] = (ep, mp, rows)
        else:
            names[split] = None
    if all(v is not None for v in names.values()):
        return {"status": "cached", "model": model_name, "dimension": dim,
                "counts": {s: len(r) for s, r in splits.items()}, "output_dir": str(out_dir.relative_to(root))}

    device = _device(config)
    try:
        processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        model = CLIPModel.from_pretrained(model_name, local_files_only=True).eval().to(device)
    except Exception as exc:
        raise RuntimeError(f"Could not load cached CLIP model '{model_name}'. Check cache/dependencies: {exc}") from exc
    actual_dim = int(model.config.projection_dim)
    if dim != actual_dim:
        dim = actual_dim
    batch_size = int(config.get("embedding_batch_size", 32))
    started = time.time()
    split_stats = {}
    for split, rows in splits.items():
        ep, mp = out_dir / f"{split}.npy", out_dir / f"{split}.metadata.json"
        if _valid_cache(ep, mp, rows, model_name, dim, model_revision):
            split_stats[split] = {"count": len(rows), "cached": True}
            continue
        done_path = out_dir / f"{split}.done.npy"
        if ep.exists():
            try:
                matrix = np.load(ep, mmap_mode="r+")
                done = np.load(done_path) if done_path.exists() else np.zeros(len(rows), dtype=bool)
                if matrix.shape != (len(rows), dim) or done.shape != (len(rows),):
                    raise ValueError("cache dimensions mismatch")
                # Reprocess only rows whose image-ID/path mapping changed in a
                # controlled manifest revision; all unaffected vectors remain cached.
                if mp.exists():
                    prior=json.loads(mp.read_text(encoding="utf-8")).get("embeddings",[])
                    if len(prior)==len(rows):
                        changed=[i for i,(old,new) in enumerate(zip(prior,rows)) if int(old.get("image_id",-1))!=int(new["image_id"]) or old.get("processed_path")!=new["processed_path"] or old.get("split")!=new["split"]]
                        done[changed]=False
                    else: done[:]=False
                else: done[:]=False
            except (OSError, ValueError):
                ep.unlink(missing_ok=True); done_path.unlink(missing_ok=True)
                matrix = np.lib.format.open_memmap(ep, mode="w+", dtype=np.float32, shape=(len(rows), dim))
                done = np.zeros(len(rows), dtype=bool)
        else:
            matrix = np.lib.format.open_memmap(ep, mode="w+", dtype=np.float32, shape=(len(rows), dim))
            done = np.zeros(len(rows), dtype=bool)
        errors = []
        pending = np.flatnonzero(~done)
        cursor = 0
        while cursor < len(pending):
            idxs = pending[cursor:cursor + batch_size]
            images = []
            valid_idxs = []
            for i in idxs:
                try:
                    with Image.open(root / rows[int(i)]["processed_path"]) as im:
                        images.append(im.convert("RGB"))
                    valid_idxs.append(int(i))
                except Exception as exc:
                    errors.append({"index": int(i), "image_id": rows[int(i)]["image_id"], "reason": str(exc)})
            if not images:
                cursor += len(idxs)
                continue
            inputs = processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            try:
                with torch.inference_mode():
                    # Use CLIP's pooled vision output and learned projection
                    # directly; Transformers releases differ in the return
                    # type of get_image_features (some return an output object).
                    vision = model.vision_model(pixel_values=pixels)
                    feats = model.visual_projection(vision.pooler_output)
                    feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
                values = feats.cpu().numpy().astype(np.float32, copy=False)
                if values.shape != (len(valid_idxs), dim):
                    raise ValueError(f"Unexpected CLIP output shape {values.shape}; expected {(len(valid_idxs),dim)}")
                matrix[valid_idxs] = values
                matrix.flush()
                done[valid_idxs] = True
                np.save(done_path, done)
            except RuntimeError as exc:
                if device.type == "cuda" and "memory" in str(exc).casefold():
                    if batch_size > 1:
                        batch_size=max(1,batch_size//2)
                        torch.cuda.empty_cache()
                        print(f"CUDA memory pressure; retrying with batch_size={batch_size}",flush=True)
                    else:
                        device=torch.device("cpu"); model.to(device); pixels=pixels.to(device)
                        torch.cuda.empty_cache()
                        print("CUDA memory pressure at batch_size=1; falling back to CPU",flush=True)
                    continue
                raise
            cursor += len(idxs)
            if cursor % (batch_size*20) < len(idxs):
                print(f"CLIP {split}: {cursor}/{len(pending)} pending; device={device}", flush=True)
        matrix.flush()
        np.save(done_path, done)
        if not done.all():
            raise RuntimeError(f"Incomplete embeddings for {split}: {int(done.sum())}/{len(done)}; errors={errors[:5]}")
        tmp_meta = out_dir / f"{split}.metadata.json.tmp"
        meta_object={"model":model_name,"model_revision":model_revision,"count":len(rows),"dimension":dim,"dtype":"float32","normalization":"L2","embeddings":rows}
        tmp_meta.write_text(json.dumps(meta_object, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_meta.replace(mp)
        ep_meta = out_dir / f"{split}.run.json"
        ep_meta.write_text(json.dumps({"model": model_name, "model_revision":model_revision,"count": len(rows), "dimension": dim,
            "dtype": "float32", "normalization": "L2", "device": str(device),"batch_size":batch_size,
            "torch_version":str(torch.__version__),"transformers_version":transformers.__version__,
            "preprocessing": "CLIPProcessor applied to processed 224x224 RGB letterboxed JPEGs",
            "errors": errors}, indent=2) + "\n", encoding="utf-8")
        split_stats[split] = {"count": len(rows), "cached": False, "errors": len(errors)}
    summary = {"status": "PASS", "model": model_name,"model_revision":model_revision, "dimension": dim, "dtype": "float32",
               "normalization": "L2", "device": str(device), "batch_size":batch_size,
               "torch_version":str(torch.__version__),"transformers_version":transformers.__version__,"elapsed_seconds": round(time.time()-started, 2),
               "splits": split_stats, "output_dir": str(out_dir.relative_to(root))}
    (out_dir / "extraction_statistics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def combined_index(root: Path, config: dict):
    """Return a mapped array and metadata in deterministic split-major order."""
    out = root / config.get("embedding_output_dir", "embeddings/image_embeddings")
    arrays, rows = [], []
    for split in ("train", "validation", "test"):
        arrays.append(np.load(out / f"{split}.npy", mmap_mode="r"))
        meta=json.loads((out / f"{split}.metadata.json").read_text(encoding="utf-8"))
        rows.extend(meta["embeddings"])
    return np.concatenate(arrays, axis=0), rows
