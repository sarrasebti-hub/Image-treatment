"""One-command orchestration of Yahia's task.

Steps
-----
1. Encode the Knowledge Base split (images + captions) with CLIP.
2. Encode the evaluation split (test by default) for held-out QC.
3. Build the FAISS vector database (image, caption and joint indices).
4. Run quality control and write ``reports/qc_report.md``.

Quick demo (500 images, ~2-3 minutes after the model is cached):
    python -m src.embeddings.pipeline --kb-limit 500 --download

Full Knowledge Base (10,000 images):
    python -m src.embeddings.pipeline --kb-limit 0 --download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_embeddings import build_split_embeddings
from .build_vector_db import build_vector_db
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    ensure_dirs,
)
from .quality_control import run_quality_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete embeddings + vector DB + QC pipeline."
    )
    parser.add_argument(
        "--kb-limit",
        type=int,
        default=0,
        help="Knowledge Base images to encode (0 = all 10,000).",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="Evaluation split images to encode (0 = all).",
    )
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing COCO images (default: enabled).",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--index-type", choices=["flat", "ivf"], default="flat")
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--k", default="1,5,10")
    parser.add_argument("--min-recall1-heldout", type=float, default=0.30)
    parser.add_argument("--min-recall1-kb", type=float, default=0.10)
    parser.add_argument("--report", default=None)
    parser.add_argument("--metrics", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    kb_limit = args.kb_limit or None
    eval_limit = args.eval_limit or None

    print("=" * 70)
    print("YAHIA TASK - EMBEDDINGS, VECTOR DATABASE & QUALITY CONTROL")
    print("=" * 70)

    build_split_embeddings(
        "knowledge_base",
        limit=kb_limit,
        download=args.download,
        model_name=args.model,
        batch_size=args.batch_size,
        seed=args.seed,
        force=args.force,
        skip_missing=args.skip_missing,
    )

    if args.eval_split:
        build_split_embeddings(
            args.eval_split,
            limit=eval_limit,
            download=args.download,
            model_name=args.model,
            batch_size=args.batch_size,
            seed=args.seed,
            force=args.force,
            skip_missing=args.skip_missing,
        )

    build_vector_db(
        split="knowledge_base",
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
        force=args.force,
    )

    if not args.skip_qc:
        run_quality_control(
            kb_split="knowledge_base",
            eval_split=args.eval_split or None,
            ks=[int(item) for item in args.k.split(",") if item.strip()],
            max_queries=args.max_queries,
            model_name=args.model,
            seed=args.seed,
            min_recall1_heldout=args.min_recall1_heldout,
            min_recall1_kb=args.min_recall1_kb,
            measure_latency_flag=not args.skip_latency,
            report_path=Path(args.report) if args.report else None,
            metrics_path=Path(args.metrics) if args.metrics else None,
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
