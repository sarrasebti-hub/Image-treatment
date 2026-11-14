from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STAGES = [
    ("split", "scripts/build_test_1000_split.py"),
    ("prepare", "scripts/prepare_test_1000.py"),
    ("metadata", "scripts/build_test_1000_metadata.py"),
    ("embeddings", "scripts/build_test_1000_embeddings.py"),
    ("verify", "scripts/verify_test_1000.py"),
    ("evaluate", "scripts/evaluate_test_1000.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", choices=[name for name, _ in STAGES], default="split")
    args = parser.parse_args()
    started = False
    for name, script in STAGES:
        if name == args.from_stage:
            started = True
        if started:
            print(f"\n=== {name} ===", flush=True)
            subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
