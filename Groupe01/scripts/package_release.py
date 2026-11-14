"""Package the NON-versioned data of release g1-v1.2 for upload as GitHub Release assets.

Run this on the machine that holds the generated embeddings / indexes (it does not generate them).

    python scripts/package_release.py                 # embeddings + FAISS indexes
    python scripts/package_release.py --with-images   # also the normalised RGB images (~1.2 GB)

Output (folder `release_assets/`, ignored by Git):
    g1-v1.2-embeddings.zip             artifacts/embeddings/**            (~400 MB)
    g1-v1.2-faiss-indexes.zip          artifacts/index/*.faiss            (~400 MB)
    g1-v1.2-test1000-embeddings.zip    experiments/test_1000_val2017/embeddings/**
    g1-v1.2-images-rgb.zip             artifacts/images_rgb/** (only with --with-images)
    SHA256SUMS.txt                     SHA-256 of every asset (publish it with the release)

GitHub limits each release asset to 2 GiB; every archive above stays well below that.
The script prints the exact `gh release create` command at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release_assets"
TAG = "g1-v1.2"
LIMIT = 2 * 1024**3 - 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_zip(name: str, patterns: list[str]) -> Path | None:
    files = sorted({p for pat in patterns for p in ROOT.glob(pat) if p.is_file()})
    if not files:
        print(f"[skip] {name}: no file matches {patterns} (not generated on this machine?)")
        return None
    target = OUT / name
    # .npy/.faiss are binary and barely compressible: store without compression (fast, same size).
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    size = target.stat().st_size
    if size > LIMIT:
        sys.exit(f"{name} is {size / 1e9:.2f} GB, above the 2 GiB release-asset limit")
    print(f"[ok]   {name}: {len(files)} files, {size / 1e6:.1f} MB")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-images", action="store_true", help="also package artifacts/images_rgb")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    assets = [
        make_zip(f"{TAG}-embeddings.zip", ["artifacts/embeddings/**/*.npy"]),
        make_zip(f"{TAG}-faiss-indexes.zip", ["artifacts/index/*.faiss"]),
        make_zip(f"{TAG}-test1000-embeddings.zip", ["experiments/test_1000_val2017/embeddings/**/*.npy"]),
    ]
    if args.with_images:
        assets.append(make_zip(f"{TAG}-images-rgb.zip", ["artifacts/images_rgb/**/*.jpg"]))
    assets = [a for a in assets if a]
    if not assets:
        sys.exit("Nothing to package: run the notebooks (04_embeddings_index) first.")

    sums = OUT / "SHA256SUMS.txt"
    sums.write_text("".join(f"{sha256(a)}  {a.name}\n" for a in assets), encoding="utf-8", newline="\n")
    print(f"\nChecksums written to {sums.relative_to(ROOT)}\n")
    names = " ".join(f'"release_assets/{a.name}"' for a in assets)
    print("Create the release (GitHub CLI):")
    print(f'  gh release create {TAG} {names} "release_assets/SHA256SUMS.txt" '
          f'--title "G1 data release {TAG}" --notes-file RELEASE_NOTES.md')


if __name__ == "__main__":
    main()
