"""Restore the three data archives of the Drive distribution, using Python only.

Existing identical files are reused. Different files are never overwritten.
Every archive and restored file is checked against the distribution manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
import uuid
import zipfile


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def inside(root, relative):
    item = PurePosixPath(relative)
    if not item.parts or item.is_absolute() or ".." in item.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe archive path: {relative}")
    target = root.joinpath(*item.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path outside destination: {relative}")
    return target


def restore(archive_root, destination, verify_only=False):
    archive_root, destination = Path(archive_root).resolve(), Path(destination).resolve()
    manifest = json.loads((archive_root / "distribution_manifest.json").read_text(encoding="utf-8"))
    restored = reused = checked = 0
    started = last_progress = time.monotonic()
    for archive in manifest["archives"]:
        filename = inside(archive_root, archive["path"])
        print(f"Verification ZIP : {archive['path']}", flush=True)
        if filename.stat().st_size != archive["bytes"] or sha256(filename) != archive["sha256"]:
            raise ValueError(f"Archive changed or incomplete: {filename}")
        expected = {f["path"]: f for f in archive["files"]}
        if len(expected) != len(archive["files"]):
            raise ValueError("Duplicate expected file paths")
        with zipfile.ZipFile(filename) as source:
            infos = source.infolist()
            if len(infos) != len(expected) or {f.filename for f in infos} != set(expected):
                raise ValueError("Archive members differ from the manifest")
            for info in infos:
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Unexpected directory or symbolic link in archive")
                record = expected[info.filename]
                target = inside(destination, info.filename)
                if info.file_size != record["bytes"]:
                    raise ValueError(f"Incorrect size: {info.filename}")
                if not verify_only and target.exists():
                    if not target.is_file() or target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
                        raise FileExistsError(f"Existing different file; refusing to overwrite: {target}")
                    reused += 1
                    continue
                digest = hashlib.sha256()
                if verify_only:
                    with source.open(info) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.restore-partial")
                    with source.open(info) as stream, temporary.open("xb") as output:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                            output.write(chunk)
                    if digest.hexdigest() != record["sha256"]:
                        raise ValueError(f"Restored checksum mismatch: {info.filename}")
                    if target.exists():
                        raise FileExistsError(f"Destination appeared during extraction: {target}")
                    temporary.replace(target)
                    restored += 1
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Checksum mismatch: {info.filename}")
                checked += 1
                now = time.monotonic()
                if now - last_progress > 10:
                    print(f"  {checked} fichiers verifies, {restored} restaures", flush=True)
                    last_progress = now
        print(f"OK : {archive['path']}", flush=True)
    result = {"status": "passed", "restored": restored, "reused": reused, "checked": checked,
              "seconds": round(time.monotonic() - started, 3), "verify_only": verify_only,
              "destination": str(destination)}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Restore Drive data without recomputing embeddings")
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--destination", type=Path, help="Default: the directory containing this script")
    parser.add_argument("--verify-only", action="store_true", help="Check ZIP contents without extraction")
    args = parser.parse_args()
    restore(args.archive_root, args.destination or Path(__file__).resolve().parent, args.verify_only)


if __name__ == "__main__":
    main()
