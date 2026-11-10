"""Run from the repository root: python -m src.cli --help."""
import argparse
import json
import sys
from pathlib import Path

from src.utils.core import load_config, path


def main():
    parser = argparse.ArgumentParser(description="COCO phase-1 data preparation; no caption generation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--offline", action="store_true", help="Load the cached encoder without network access")
    parser.add_argument("--log-file", help="Write stdout/stderr directly to a UTF-8 log file")
    parser.add_argument("command", choices=["download-annotations", "download-images", "download-nlp", "download-model", "splits", "extend-test", "audit", "metadata", "eda", "embeddings", "index", "qc", "smoke", "retrieve", "validate-retrieval", "report"])
    parser.add_argument("--force", action="store_true", help="Explicit replacement of frozen splits or embedding cache")
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--image-path", help="Path to a query image, relative to the repository or absolute")
    parser.add_argument("--top-k", type=int, help="Override retrieval.top_k for this query")
    parser.add_argument("--download-workers", type=int, help="Override download worker count for this run (recorded in manifest)")
    parser.add_argument("--output", help="Output directory for manual retrieval")
    parser.add_argument("--show", action="store_true", help="Open the saved retrieval gallery in the default browser")
    parser.add_argument("--split", choices=["knowledge_base", "validation", "test"], default="validation")
    parser.add_argument("--kind", choices=["image", "text"], default="image")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_offline"] = args.offline
    cfg["_download_workers"] = args.download_workers
    if args.log_file:
        log_file = Path(args.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_file.open("w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log_stream
    try:
        if args.command == "download-annotations":
            from src.data.download import download_annotations
            result = download_annotations(cfg)
        elif args.command == "download-model":
            from src.embeddings.encode import Encoder
            from src.utils.core import stamp, write_json
            encoder = Encoder(cfg)
            result = {**stamp(cfg), "model": cfg["embedding"]["model"],
                      "resolved_revision": encoder.resolved_revision, "device": encoder.device,
                      "cpu_threads": encoder.torch.get_num_threads(), "status": "pretrained_weights_loaded"}
            write_json(path(cfg, "reports") / "model_preparation.json", result)
        elif args.command == "download-nlp":
            import nltk
            destination = path(cfg, "cache") / "nltk_data"
            destination.mkdir(parents=True, exist_ok=True)
            if not nltk.download(cfg["analysis"]["pos_resource"], download_dir=str(destination), raise_on_error=True):
                raise RuntimeError("NLTK download failed")
            result = {"resource": cfg["analysis"]["pos_resource"], "directory": str(destination)}
        elif args.command == "report":
            from src.utils.report import phase_report
            result = phase_report(cfg)
        elif args.command == "index":
            from src.indexing.faiss_index import build_indexes
            result = build_indexes(cfg)
        elif args.command == "eda":
            from src.exploration.eda import run_eda
            result = run_eda(cfg)
        elif args.command == "embeddings":
            from src.embeddings.encode import generate_embeddings
            result = generate_embeddings(cfg, force=args.force)
        elif args.command == "validate-retrieval":
            from src.indexing.retrieval import validation_retrieval
            result = validation_retrieval(cfg, top_k=args.top_k)
        elif args.command == "retrieve":
            from src.indexing.retrieval import manual_retrieval
            result = manual_retrieval(cfg, image_id=args.image_id, image_path=args.image_path,
                                      top_k=args.top_k, output=args.output, show=args.show)
        else:
            from src.data.coco import CocoData
            data = CocoData(cfg)
            if args.command == "splits":
                from src.data.splits import create_splits
                result = create_splits(cfg, data, args.force)
            elif args.command == "extend-test":
                from src.data.splits import extend_test
                result = extend_test(cfg, data)
            elif args.command == "download-images":
                from src.data.download import download_images
                result = download_images(cfg, data)
            elif args.command == "audit":
                from src.data.audit import audit
                result = audit(cfg, data)
            elif args.command == "metadata":
                from src.preprocessing.metadata import prepare_metadata
                result = prepare_metadata(cfg, data)
            elif args.command == "qc":
                from src.preprocessing.quality import quality_control
                result = quality_control(cfg, data)
            elif args.command == "smoke":
                from src.indexing.faiss_index import smoke_test
                result = smoke_test(cfg, data, args.image_id, args.split, args.kind)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
