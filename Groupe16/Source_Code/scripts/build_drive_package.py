"""Package phase 1 as visible source/reports plus three separate large ZIPs.

No inference, new sampling, downloads, training, or changes to original data.
Run from the existing prepared project: python scripts/build_drive_package.py
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.core import file_hash

DEST = ROOT / "releases/Projet_ITI_Drive_v2"
SKIP = {"__pycache__", ".ipynb_checkpoints", ".git"}

README = r"""# Mini-projet ITI - phase 1

Ce dossier est organise pour Google Drive : le code et le rapport sont directement
consultables, et les gros fichiers sont regroupes dans trois archives ZIP.

## Consulter le travail

- `output/pdf/Rapport_phase1_FR.pdf` : rapport explique en francais.
- `output/pdf/Phase1_report_EN.pdf` : rapport en anglais.
- `src/` : code Python, organise selon les etapes du projet.
- `notebooks/` : notebooks.
- `reports/figures/`, `reports/tables/` : statistiques et visualisations.
- `METHODOLOGIE_ET_COMMANDES.md` : documentation technique complete.
- `docs/project/` : cahier des charges et precisions transmises.

## Les trois archives de donnees

| Archive | Contenu | Destination apres restauration |
|---|---|---|
| `data/01_images_annotations.zip` | Photos selectionnees et annotations COCO originales | `data/raw/` |
| `models/02_clip_nltk.zip` | Poids CLIP, processeur et tokenizer, ressource NLTK | `data/interim/cache/` |
| `artifacts/03_metadata_embeddings_faiss.zip` | Metadonnees, embeddings image/texte, index FAISS et correspondances | `artifacts/` |

Les listes gelees `artifacts/splits/` restent visibles, hors ZIP. Les ZIP
contiennent deja leurs chemins relatifs : **ne pas les extraire manuellement
dans un sous-dossier portant le nom de l'archive**.

## Tester apres telechargement du dossier

1. Telecharger le dossier complet depuis Drive, puis extraire le ZIP cree par Drive.
2. Ouvrir le dossier contenant ce README dans VS Code, puis ouvrir un terminal PowerShell.
3. Installer Python 3.12 x64 si necessaire. Les bibliotheques Python ne sont pas incluses
   dans cette distribution. L'installation suivante necessite Internet.
4. Executer les commandes une par une ; s'arreter si une commande echoue :

```powershell
py -3.12 restaurer_donnees.py
py -3.12 -m venv imp
.\imp\Scripts\python.exe -m pip install -r requirements.txt
.\imp\Scripts\python.exe -m src.cli --offline retrieve --image-id 490 --top-k 5 --show
```

La restauration remet les fichiers au bon endroit et controle leurs empreintes.
Elle ne telecharge pas les photos, ne recalcule pas les embeddings et ne modifie
pas les fichiers existants s'ils sont differents. Si les fichiers identiques sont
deja presents, ils sont reutilises. La premiere restauration prend du temps pour
extraire les fichiers ; les demonstrations suivantes n'en ont plus besoin.

Autres essais :

```powershell
.\imp\Scripts\python.exe -m src.cli --offline retrieve --image-id 133 --top-k 10 --show
.\imp\Scripts\python.exe -m src.cli --offline retrieve --image-path "C:\photos\ma_photo.jpg" --top-k 5 --show
```

Remplacer le chemin de la derniere commande par une vraie photo. Les resultats
sont sauvegardes dans `reports/retrieval/` : galerie HTML, image PNG, JSON et CSV.
Pour refaire le controle complet des donnees et artefacts :

```powershell
.\imp\Scripts\python.exe -m src.cli --offline qc
```

La verification de ce dossier utilise l'environnement Python deja present sur
le PC de preparation ; elle ne prouve pas une installation propre sur tout PC.
Les versions de l'execution scientifique sont dans les manifestes et
`reports/runtime_direct_requirements.txt`.

## Etat scientifique exact

- 10 000 images de base et 100 images de validation distinctes, issues de train2017.
- Test etendu de 1 000 images val2017 prepare et audite.
- Premier test de 100 images conserve dans les 1 000, avec statistiques separees.
- 11 100 images distinctes au total ; liste commune de l'enseignante non fournie.
- References humaines : confiance=1 et seuil=1 explicites ; heuristiques separees.
- 10 000 embeddings image et 50 000 embeddings texte CLIP ; index FAISS sauvegardes.
- Objets issus des annotations COCO ; aucun detecteur visuel execute.
- Aucune generation finale de captions, aucun entrainement ou evaluation finale du RAG.

Cette organisation change le rangement pour le partage, pas les resultats.
L'ancienne distribution autonome Windows avec Python inclus reste un autre paquet.
Elle n'est pas necessaire pour consulter ou utiliser cette distribution.
"""


def visible_files():
    for name in ("src", "configs", "docs", "examples", "notebooks", "reports", "tests", "licenses", "output/pdf", "artifacts/splits"):
        for p in (ROOT / name).rglob("*"):
            if name == "reports" and "releases" in p.relative_to(ROOT / name).parts:
                continue
            if p.is_file() and not SKIP.intersection(p.relative_to(ROOT).parts) and p.name != ".gitkeep":
                yield p
    for name in ("requirements.txt", "environment.yml", ".gitattributes"):
        yield ROOT / name
    if (ROOT / "TESTER.cmd").is_file():
        yield ROOT / "TESTER.cmd"
    for p in (ROOT / "scripts").glob("*.py"):
        yield p


def archives():
    records = []
    for split in ("knowledge_base", "validation", "test"):
        records.extend(json.loads(line) for line in (ROOT / f"artifacts/metadata/{split}.jsonl").read_text(encoding="utf-8").splitlines())
    images = sorted({ROOT / "data/raw" / r["coco_source"] / r["file_name"] for r in records})
    annotations = sorted((ROOT / "data/raw/annotations").glob("*.json"))
    assert len(images) == 11100 and len(annotations) == 4
    yield "data/01_images_annotations.zip", images + annotations
    em = json.loads((ROOT / "artifacts/embeddings/manifest.json").read_text(encoding="utf-8"))
    snapshot = ROOT / "data/interim/cache/huggingface/models--openai--clip-vit-base-patch32/snapshots" / em["resolved_revision"]
    model = [p for p in snapshot.rglob("*") if p.is_file()]
    model += [p for p in (ROOT / "data/interim/cache/nltk_data").rglob("*") if p.is_file() and p.suffix != ".zip"]
    assert any(p.name == "pytorch_model.bin" for p in model)
    yield "models/02_clip_nltk.zip", sorted(model)
    artifacts = [p for folder in ("metadata", "embeddings", "indexes") for p in (ROOT / "artifacts" / folder).glob("*") if p.is_file() and p.name != ".gitkeep"]
    artifacts += [p for p in (ROOT / "reports/releases").rglob("*") if p.is_file()]
    yield "artifacts/03_metadata_embeddings_faiss.zip", sorted(artifacts)


def main():
    if DEST.exists():
        raise FileExistsError(f"Refusing to replace an existing distribution: {DEST}")
    DEST.mkdir(parents=True)
    start = time.monotonic()
    visible = []
    for src in visible_files():
        relative = src.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        visible.append({"path": relative.as_posix(), "bytes": src.stat().st_size, "sha256": file_hash(target)})
    shutil.copy2(ROOT / "README.md", DEST / "METHODOLOGIE_ET_COMMANDES.md")
    shutil.copy2(ROOT / "scripts/restore_drive_data.py", DEST / "restaurer_donnees.py")
    (DEST / "README.md").write_text(README, encoding="utf-8")
    (DEST / "LIRE_MOI.txt").write_text(
        "PROJET ITI - DOSSIER POUR GOOGLE DRIVE\n\n"
        "Code, rapport et statistiques : directement consultables.\n"
        "Photos, modele et vecteurs : trois ZIP dans data/, models/ et artifacts/.\n\n"
        "Pour consulter le rapport : output/pdf/Rapport_phase1_FR.pdf\n"
        "Pour tester : suivre README.md (restaurer_donnees.py puis installation Python).\n"
        "Il n'est pas necessaire de recalculer les embeddings.\n\n"
        "Pour partager : importer ce dossier entier dans Google Drive, sans le zipper de nouveau.\n"
        "Dans Drive : Nouveau > Importer un dossier > selectionner Projet_ITI_Drive_v2.\n",
        encoding="utf-8")
    inventories = []
    for archive_name, files in archives():
        target = DEST / archive_name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".zip.partial")
        inventory = []
        print(f"Creation {archive_name}: {len(files)} fichiers", flush=True)
        progress = time.monotonic()
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for i, p in enumerate(files, 1):
                relative = p.relative_to(ROOT).as_posix()
                kind = zipfile.ZIP_STORED if p.suffix.lower() in (".jpg", ".jpeg", ".png") else zipfile.ZIP_DEFLATED
                archive.write(p, relative, compress_type=kind, compresslevel=1)
                inventory.append({"path": relative, "bytes": p.stat().st_size, "sha256": file_hash(p)})
                if time.monotonic() - progress > 10:
                    print(f"  {i}/{len(files)} fichiers archives", flush=True)
                    progress = time.monotonic()
        partial.replace(target)
        result = {"path": archive_name, "bytes": target.stat().st_size, "sha256": file_hash(target), "files": inventory}
        inventories.append(result)
        print(f"Termine : {archive_name} ({result['bytes'] / 1e6:.1f} MB)", flush=True)
    manifest = {"schema_version": 1, "scope": "Existing phase-1 release, reorganized for Google Drive",
                "frozen_sizes": {"knowledge_base": 10000, "validation": 100, "test": 1000},
                "comparison_test_100": "Original 100 nested in test1000; separate statistics; not counted twice",
                "updated_test_target": 1000, "extended_test_prepared": True,
                "python_runtime_bundled": False, "archives": inventories, "visible_files": visible,
                "model_recomputed": False, "embeddings_recomputed": False}
    (DEST / "distribution_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"folder": str(DEST), "archive_bytes": sum(r["bytes"] for r in inventories),
               "visible_files": len(visible), "archives": [{k: r[k] for k in ("path", "bytes", "sha256")} for r in inventories],
               "elapsed_seconds": round(time.monotonic()-start, 3)}
    (ROOT / "releases/drive_package_build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
