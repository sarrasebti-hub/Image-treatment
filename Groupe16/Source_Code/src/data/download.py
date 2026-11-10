"""Explicit downloads only: annotations first, then selected image IDs."""
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from zipfile import ZipFile

from PIL import Image
from tqdm import tqdm

from src.data.splits import SPLIT_SOURCES, load_splits
from src.utils.core import file_hash, path, stamp, write_json


def fetch(url, target, cfg):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(cfg["download"]["retries"]):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "University-COCO-Phase1/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                response = urlopen(Request(url, headers=headers), timeout=cfg["download"]["timeout_seconds"])
            except HTTPError as exc:
                if exc.code != 416 or not offset:
                    raise
                # A previously complete/stale .part can make its Range unsatisfiable.
                # Fetch a fresh full response instead of retrying the same bad range.
                offset = 0
                response = urlopen(Request(url, headers={"User-Agent": "University-COCO-Phase1/1"}),
                                   timeout=cfg["download"]["timeout_seconds"])
            with response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected download range")
                with temporary.open("ab" if append else "wb") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                content_length = response.headers.get("Content-Length")
                expected = (offset if append else 0) + int(content_length) if content_length is not None else None
                if expected is not None and temporary.stat().st_size != expected:
                    raise OSError("Incomplete response body; partial file retained for resume")
            temporary.replace(target)
            return
        except (OSError, TimeoutError):
            if attempt + 1 == cfg["download"]["retries"]:
                raise
            time.sleep(min(2 ** attempt, 8))


def download_annotations(cfg):
    directory = path(cfg, "annotations")
    names = [f"{kind}_{source}.json" for kind in ("captions", "instances") for source in ("train2017", "val2017")]
    if all((directory / name).is_file() for name in names):
        return {"status": "already_present"}
    archive = path(cfg, "raw") / "annotations_trainval2017.zip"
    if not archive.is_file():
        fetch(cfg["download"]["annotations_url"], archive, cfg)
    # Extract only explicitly selected members; no arbitrary paths from the ZIP.
    directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        for name in names:
            target = directory / name
            if target.exists():
                continue
            temporary = target.with_suffix(".json.part")
            with bundle.open(f"annotations/{name}") as source, temporary.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            temporary.replace(target)
    result = {**stamp(cfg), "url": cfg["download"]["annotations_url"],
              "archive_sha256": file_hash(archive),
              "files": {name: file_hash(directory / name) for name in names},
              "integrity_note": "Local SHA256 and ZIP CRC; no publisher-provided checksum verified."}
    write_json(path(cfg, "reports") / "annotation_download.json", result)
    return result


def download_images(cfg, data):
    splits = load_splits(cfg)
    results = []
    tasks = []
    for split, ids in splits.items():
        source = SPLIT_SOURCES[split]
        tasks.extend((split, source, image_id) for image_id in ids)

    def download_one(task):
        split, source, image_id = task
        result = {"image_id": image_id, "source": source, "split": split}
        try:
            image = data.captions[source].imgs[image_id]
            target = data.image_path(source, image)
            if target.exists():
                return {**result, "status": "already_present"}
            url = f"{cfg['download']['image_base_url'].rstrip('/')}/{source}/{image['file_name']}"
            fetch(url, target, cfg)
            with Image.open(target) as img:
                img.verify()
            return {**result, "status": "downloaded"}
        except Exception as exc:
            return {**result, "status": "error", "error": str(exc)}

    workers = cfg.get("_download_workers") or cfg["download"].get("workers", 4)
    if type(workers) is not int or workers <= 0:
        raise ValueError("Download workers must be a positive integer")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="COCO images", mininterval=5):
            try:
                results.append(future.result())
            except Exception as exc:
                # Record a worker exception before waiting for other downloads.
                write_json(path(cfg, "reports") / "image_download_worker_error.json", {"error": repr(exc)})
                for pending in futures:
                    pending.cancel()
                raise
            if len(results) % 100 == 0:
                write_json(path(cfg, "reports") / "image_download_progress.json", {
                    "completed": len(results), "total": len(tasks),
                    "errors": sum(row["status"] == "error" for row in results)})
    results.sort(key=lambda row: (row["split"], row["image_id"]))
    report = {**stamp(cfg), "events": results, "workers": workers,
              "counts": dict(Counter(row["status"] for row in results)),
              "image_base_url": cfg["download"]["image_base_url"]}
    write_json(path(cfg, "reports") / "image_download.json", report)
    if any(row["status"] == "error" for row in results):
        raise ValueError("Image downloads failed. See reports/image_download.json; files retained for audit.")
    return report
