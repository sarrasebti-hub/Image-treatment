"""Exécute les notebooks dans l'ordre (sans interface) et sauvegarde les versions exécutées.
  python tools/run_notebooks.py                       # vraies données (config.yaml), tous les notebooks 00..09
  python tools/run_notebooks.py --config config_test.yaml --skip 00_download_annotations 01b_download_images   # test synthétique sans réseau
Les notebooks d'origine restent propres (sans sorties) ; les copies exécutées vont dans executed_notebooks/.
"""
import argparse, os, shutil, stat, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["00_download_annotations", "01_make_splits", "01b_download_images", "02_preprocess_images", "03_captions_metadata", "04_embeddings_index", "05_eda",
         "06_select_config", "07_qc_leakage", "08_freeze", "09_demo_retrieval"]   # 03b_detect_objects : optionnel, à lancer à la main

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml"); ap.add_argument("--skip", nargs="*", default=[])
ap.add_argument("--only", nargs="*"); ap.add_argument("--out", default="executed_notebooks"); ap.add_argument("--timeout", type=int, default=36000)
a = ap.parse_args()
os.environ["RAG_CONFIG"] = a.config
out = ROOT / a.out; out.mkdir(exist_ok=True)
for art in ROOT.glob("artifacts*"):                       # déverrouille une release déjà figée avant de relancer
    for f in art.rglob("*"):
        if f.is_file():
            f.chmod(f.stat().st_mode | stat.S_IWUSR)
for name in (a.only or ORDER):
    if name in a.skip:
        print("skip", name); continue
    t = time.time()
    nb = nbformat.read(ROOT / f"{name}.ipynb", 4)
    try:
        NotebookClient(nb, timeout=a.timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        status = "OK"
    except CellExecutionError as e:
        status = "ERREUR"; print(str(e)[-1800:])
    nbformat.write(nb, out / f"{name}.ipynb")
    print(f"{name:26s} {status}  ({time.time() - t:.0f}s)")
    if status != "OK":
        sys.exit(1)
