"""Run the complete dataset regeneration and validation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "01_src"
NOTEBOOK_ROOT = SRC_ROOT / "notebooks"


def run_command(label: str, command: list[str], env: dict[str, str]) -> None:
    """Run one pipeline stage and stop if it fails."""

    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def run_python_script(
    label: str,
    script: Path,
    env: dict[str, str],
) -> None:
    """Run a repository Python script with the active interpreter."""

    run_command(label, [sys.executable, str(script)], env)


def run_notebook(
    label: str,
    notebook: Path,
    output_dir: Path,
    env: dict[str, str],
) -> None:
    """Execute a notebook without modifying the source notebook."""

    run_command(
        label,
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=-1",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output",
            f"executed_{notebook.name}",
            "--output-dir",
            str(output_dir),
            str(notebook),
        ],
        env,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate and validate the configured Phase 1 dataset release."
        )
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help=(
            "Run split generation, extraction, and validation only. "
            "Do not regenerate preprocessing or downstream artifacts."
        ),
    )
    return parser


def main() -> int:
    """Execute all configured pipeline stages."""

    args = build_parser().parse_args()

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not existing_pythonpath
        else os.pathsep.join([str(SRC_ROOT), existing_pythonpath])
    )

    with tempfile.TemporaryDirectory(
        prefix="image-captioning-rag-pipeline-"
    ) as temporary_directory:
        notebook_output_dir = Path(temporary_directory)

        run_python_script(
            "1/8 Generating dataset splits",
            SRC_ROOT / "data" / "split_dataset.py",
            env,
        )
        run_python_script(
            "2/8 Extracting annotations and images",
            SRC_ROOT / "data" / "extract_annotations.py",
            env,
        )
        run_python_script(
            "3/8 Validating extracted dataset",
            SRC_ROOT / "data" / "validate_dataset.py",
            env,
        )

        if args.skip_notebooks:
            print("\nNotebook stages skipped by request.")
            return 0

        run_notebook(
            "4/8 Running image and caption preprocessing",
            NOTEBOOK_ROOT / "02_preprocessing_experiments.ipynb",
            notebook_output_dir,
            env,
        )
        run_python_script(
            "5/8 Revalidating processed dataset",
            SRC_ROOT / "data" / "validate_dataset.py",
            env,
        )
        run_notebook(
            "6/8 Generating embeddings and knowledge-base artifacts",
            NOTEBOOK_ROOT / "03_embedding_experiments.ipynb",
            notebook_output_dir,
            env,
        )
        run_notebook(
            "7/8 Running knowledge-base quality control",
            NOTEBOOK_ROOT / "04_knowledge_base_quality_control.ipynb",
            notebook_output_dir,
            env,
        )
        run_notebook(
            "8/8 Running retrieval validation",
            NOTEBOOK_ROOT / "05_retrieval_pipeline_validation.ipynb",
            notebook_output_dir,
            env,
        )

    print("\nFull pipeline regeneration completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
