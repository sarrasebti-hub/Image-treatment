"""Build a self-contained Windows x64 ZIP from the verified local phase-1 run.

No dependency installation, model inference or embedding recomputation takes place.
Dependencies are copied from their installed distribution RECORDs, including licenses.
"""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "dist/portable_runtime/runtime"
SEEDS = ["torch", "torchvision", "transformers", "faiss-cpu", "numpy", "Pillow", "PyYAML",
         "matplotlib", "pycocotools", "nltk", "huggingface-hub", "tqdm", "nbformat"]
EXCLUDE_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def distributions():
    environment = {**default_environment(), "extra": ""}
    pending = SEEDS.copy()
    result = {}
    while pending:
        distribution = metadata.distribution(pending.pop())
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in result:
            continue
        result[name] = distribution
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate(environment):
                pending.append(requirement.name)
    return result


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, target)


def prepare_runtime():
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Build with the audited Windows x64 Python 3.12 interpreter")
    base = Path(sys.base_prefix)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for name in ("python.exe", "pythonw.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"):
        copy_file(base / name, RUNTIME / name)
    # App-local C++ runtime dependencies used by the installed CPU wheels.
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    for name in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll", "vcomp140.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll"):
        if (system32 / name).is_file():
            copy_file(system32 / name, RUNTIME / name)
    packages = distributions()
    versions = {name: distribution.version for name, distribution in sorted(packages.items())}
    version_file = RUNTIME.parent / "runtime_packages.json"
    if version_file.is_file() and (RUNTIME / "python312._pth").is_file():
        if json.loads(version_file.read_text(encoding="utf-8")) == versions:
            print("Reusing the complete, already-tested portable runtime", flush=True)
            return versions
    for directory in ("Lib", "DLLs"):
        for current, directories, filenames in os.walk(base / directory):
            directories[:] = [d for d in directories if d not in EXCLUDE_PARTS | {"site-packages", "test", "tests", "idlelib", "tkinter"}]
            for name in filenames:
                source = Path(current) / name
                copy_file(source, RUNTIME / source.relative_to(base))
    destinations = set()
    for name, distribution in sorted(packages.items()):
        site = Path(distribution.locate_file("")).resolve()
        count = 0
        for record in distribution.files or []:
            source = Path(distribution.locate_file(record)).resolve()
            if not source.is_file() or not source.is_relative_to(site):
                continue
            relative = source.relative_to(site)
            if EXCLUDE_PARTS.intersection(relative.parts) or source.suffix == ".pyc":
                continue
            target = RUNTIME / "Lib/site-packages" / relative
            if target not in destinations:
                copy_file(source, target)
                destinations.add(target)
                count += 1
        print(f"Runtime {name}=={distribution.version}: {count} files", flush=True)
    (RUNTIME / "python312._pth").write_text(".\nDLLs\nLib\nLib/site-packages\n..\nimport site\n", encoding="utf-8")
    (RUNTIME.parent / "runtime_packages.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    return versions


def project_files():
    for folder in ("src", "configs", "docs", "examples", "notebooks", "tests", "reports", "scripts", "licenses"):
        for file in (ROOT / folder).rglob("*"):
            if file.is_file() and not EXCLUDE_PARTS.intersection(file.relative_to(ROOT).parts):
                yield file
    for name in ("README.md", "requirements.txt", "environment.yml"):
        yield ROOT / name
    for folder in ("splits", "metadata", "embeddings", "indexes"):
        for file in (ROOT / "artifacts" / folder).glob("*"):
            if file.is_file() and file.name != ".gitkeep":
                yield file
    for file in (ROOT / "data/raw/annotations").glob("*.json"):
        yield file
    for split, source in (("knowledge_base", "train2017"), ("validation", "train2017"), ("test", "val2017")):
        records = [json.loads(line) for line in (ROOT / f"artifacts/metadata/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in records:
            yield ROOT / "data/raw" / source / row["file_name"]
    model = "models--openai--clip-vit-base-patch32"
    revision = json.loads((ROOT / "artifacts/embeddings/manifest.json").read_text())["resolved_revision"]
    for file in (ROOT / "data/interim/cache/huggingface" / model / "snapshots" / revision).rglob("*"):
        if file.is_file():
            yield file
    for file in (ROOT / "data/interim/cache/nltk_data").rglob("*"):
        if file.is_file() and file.suffix != ".zip":
            yield file


QUICKSTART = """COCO / CLIP / FAISS - DEMONSTRATION PRETE A UTILISER
====================================================
Windows 10/11, 64 bits. Python et bibliotheques inclus.

1. Extraire TOUT le ZIP dans un dossier accessible en ecriture.
2. Double-cliquer sur TESTER.cmd.
3. Attendre le chargement de CLIP; la galerie des 5 voisins s'ouvre.

Aucune installation, aucun telechargement, aucun recalcul des 10 000 vecteurs.
Les 10 000 photos de la base, 100 de validation et 100 de test provisoire sont incluses.

Photo personnelle : glisser le fichier photo sur TESTER.cmd.
Terminal PowerShell dans ce dossier :
  .\\TESTER.cmd --image-id 490 --top-k 5
  .\\TESTER.cmd --image-id 133 --top-k 10
  .\\TESTER.cmd "C:\\chemin\\ma photo.jpg" --top-k 10

Le nom 000000000133.jpg correspond a l'ID COCO 133, pas au numero de position.
Choisir visuellement les images dans data/raw/train2017/.
Les resultats se trouvent sous reports/retrieval/manual_.../.

Le premier lancement charge le modele en memoire et peut prendre quelques secondes.
Les captions affichees sont les references COCO existantes, aucune caption finale n'est generee.
Ce paquet est pour Windows x64, pas pour macOS/Linux ou Windows ARM natif.
Code scientifique et definitions : README.md et reports/.
"""


def build_zip():
    import hashlib
    from time import perf_counter
    output = ROOT / "releases/COCO_Phase1_Windows_x64.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    versions = prepare_runtime()
    files = {file.relative_to(ROOT).as_posix(): file for file in project_files()}
    for file in RUNTIME.rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts:
            files["runtime/" + file.relative_to(RUNTIME).as_posix()] = file
    plan = {name: {"bytes": file.stat().st_size} for name, file in sorted(files.items())}
    total = sum(row["bytes"] for row in plan.values())
    print(f"ZIP: {len(files)} files, {total / 1e9:.2f} GB uncompressed", flush=True)
    prefix = "COCO_Phase1_Windows/"
    started = perf_counter()
    temporary = output.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for number, (name, file) in enumerate(sorted(files.items()), 1):
            compression = zipfile.ZIP_STORED if file.suffix.lower() in (".jpg", ".png", ".jpeg", ".pdf") else zipfile.ZIP_DEFLATED
            archive.write(file, prefix + name, compress_type=compression, compresslevel=1)
            if number % 1000 == 0:
                print(f"Packed {number}/{len(files)} files", flush=True)
        archive.write(ROOT / "scripts/portable_launcher.py", prefix + "portable_launcher.py")
        archive.writestr(prefix + "TESTER.cmd", '@echo off\r\ncd /d "%~dp0"\r\n"%~dp0runtime\\python.exe" -I "%~dp0portable_launcher.py" %*\r\nif errorlevel 1 pause\r\n')
        archive.writestr(prefix + "LIRE_MOI.txt", QUICKSTART)
        archive.writestr(prefix + "runtime_packages.json", json.dumps(versions, indent=2))
        archive.writestr(prefix + "bundle_manifest.json", json.dumps({"platform": "Windows x64", "python": "3.12.10",
                         "images": {"knowledge_base": 10000, "validation": 100, "test_provisional": 100},
                         "model_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "files": plan}, indent=2))
    temporary.replace(output)
    sha = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    output.with_suffix(".zip.sha256").write_text(sha.hexdigest() + "  " + output.name + "\n", encoding="ascii")
    print(json.dumps({"zip": str(output), "bytes": output.stat().st_size, "sha256": sha.hexdigest(),
                      "seconds": perf_counter() - started}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-only", action="store_true")
    options = parser.parse_args()
    if options.runtime_only:
        prepare_runtime()
    else:
        build_zip()
