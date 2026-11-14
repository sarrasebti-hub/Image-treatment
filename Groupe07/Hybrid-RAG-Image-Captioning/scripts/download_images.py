
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

SPLITS_FILE = DATASET_DIR / "dataset_splits.json"

TRAIN_DIR = DATASET_DIR / "images_train"
VAL_DIR = DATASET_DIR / "images_val"

BASE_URL = "https://images.cocodataset.org"

MAX_WORKERS = 8


def download_one(image_id, folder, split):
    filename = f"{image_id:012d}.jpg"
    destination = folder / filename

    if destination.exists() and destination.stat().st_size > 0:
        return True

    url = f"{BASE_URL}/{split}/{filename}"

    try:
        result = subprocess.run(
            [
                "curl.exe",
                "-k",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--retry", "3",
                "-o", str(destination),
                url
            ],
            capture_output=True,
            text=True,
            timeout=90
        )

        if result.returncode == 0 and destination.exists():
            if destination.stat().st_size > 0:
                return True

        if destination.exists():
            destination.unlink()

        return False

    except Exception:
        if destination.exists():
            destination.unlink()
        return False


def download_split(image_ids, folder, split_name, source_folder):
    print(f"\nDownloading {split_name}...")

    folder.mkdir(parents=True, exist_ok=True)
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                download_one, image_id, folder, source_folder
            ): image_id
            for image_id in image_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            image_id = futures[future]

            try:
                if not future.result():
                    failed.append(image_id)
            except Exception:
                failed.append(image_id)

    print(f"{split_name}: {len(image_ids) - len(failed)} downloaded")
    print(f"Failed: {len(failed)}")

    return failed


def main():
    with open(SPLITS_FILE, "r", encoding="utf-8") as file:
        splits = json.load(file)

    train_and_validation = sorted(
        set(splits["train"]) | set(splits["validation"])
    )

    failed_train = download_split(
        train_and_validation,
        TRAIN_DIR,
        "Train + Validation",
        "train2017"
    )

    failed_test = download_split(
        splits["test"],
        VAL_DIR,
        "Test",
        "val2017"
    )

    failures = {
        "train2017": failed_train,
        "val2017": failed_test
    }

    with open(
        DATASET_DIR / "download_failures.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(failures, file, indent=4)

    print("\nImage download completed!")
    print(f"Train/Validation failures: {len(failed_train)}")
    print(f"Test failures: {len(failed_test)}")


if __name__ == "__main__":
    main()
