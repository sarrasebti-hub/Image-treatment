"""Entry point shipped inside the self-contained Windows demonstration ZIP."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "data/interim/cache/matplotlib")
os.environ["MPLBACKEND"] = "Agg"
os.environ["NLTK_DATA"] = str(ROOT / "data/interim/cache/nltk_data")


def main():
    if sys.argv[1:2] == ["--cli"]:
        sys.argv = ["src.cli", "--offline", *sys.argv[2:]]
    else:
        parser = argparse.ArgumentParser(description="Recherche CLIP/FAISS prete a utiliser, sans installation.")
        parser.add_argument("photo", nargs="?", help="Chemin d'une photo (ou glisser-deposer sur TESTER.cmd)")
        parser.add_argument("--image-id", type=int, help="Identifiant COCO disponible dans le paquet")
        parser.add_argument("--top-k", type=int, default=5, help="Nombre de voisins, par defaut 5")
        parser.add_argument("--no-show", action="store_true", help="Enregistrer les resultats sans ouvrir le navigateur")
        parser.add_argument("--runtime-info", action="store_true", help="Verifier le Python et les bibliotheques inclus")
        args = parser.parse_args()
        if args.runtime_info:
            import faiss
            import numpy
            import torch
            import transformers
            import site
            modules = {m.__name__: str(Path(m.__file__).resolve()) for m in (faiss, numpy, torch, transformers)}
            if not all(Path(p).is_relative_to(ROOT / "runtime") for p in modules.values()):
                raise RuntimeError("Une bibliotheque externe au paquet a ete chargee")
            print(json.dumps({"python": sys.version, "executable": sys.executable,
                              "isolated": bool(sys.flags.isolated), "user_site": site.ENABLE_USER_SITE,
                              "modules": modules, "sys_path": sys.path}, indent=2))
            return
        if args.photo and args.image_id is not None:
            parser.error("Choisis une photo OU un ID, pas les deux")
        if args.top_k <= 0:
            parser.error("--top-k doit etre positif")
        selection = ["--image-path", str(Path(args.photo).resolve())] if args.photo else ["--image-id", str(
            args.image_id if args.image_id is not None else json.loads(
                (ROOT / "artifacts/splits/validation_ids.json").read_text(encoding="utf-8"))[0])]
        print("Recherche dans les 10 000 images deja preparees. Chargement de CLIP...", flush=True)
        sys.argv = ["src.cli", "--offline", "retrieve", *selection, "--top-k", str(args.top_k)]
        if not args.no_show:
            sys.argv.append("--show")
    from src.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
