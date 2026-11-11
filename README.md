# Mini-projet — Phase 1 : exploration et prétraitement (COCO)

## Installation (terminal VS Code, à la racine du projet)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   |   source .venv/bin/activate   (Linux/Mac)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
VS Code : `Ctrl+Shift+P` -> *Python: Select Interpreter* -> `.venv`.
Pas de GPU ? Rien à faire : le code bascule automatiquement sur CPU.

## Ordre d'exécution
| # | Commande | Rôle | Durée (CPU) |
|---|---|---|---|
| 0 | `python scripts/00_download.py` (ou `--skip-images` si le prof fournit les images) | annotations + images | 10-30 min |
| 1 | `python scripts/01_audit_cleaning.py` | audit, doublons, rapport | ~2 min |
| 2 | `python scripts/02_splits.py` | splits KB/val/test + anti-fuite | <1 min |
| 3 | `python scripts/03_eda.py` | stats + graphiques PNG | ~2 min |
| 4 | `python scripts/04_image_preprocessing.py` | diagnostic + démo prétraitement | ~1 min |
| 5 | `python scripts/05_captions_metadata.py` | captions brutes/nettoyées + métadonnées | <1 min |
| 6 | `python scripts/06_embeddings.py` | embeddings CLIP | 20-40 min (reprise auto) |
| 7 | `python scripts/07_faiss_index.py` | index FAISS + démo | <1 min |
| 8 | `python scripts/08_quality_control.py` | contrôle qualité (PASS/FAIL) | <1 min |
| 9 | `python scripts/09_report.py` | rapport final `results/reports/phase1_report.md` | <1 s |

Test rapide du pipeline sur un petit échantillon (PowerShell : `$env:N_KB=500`) :
`N_KB=500 N_VAL=50 N_TEST=50 N_CANDIDATES=600` avant les commandes. Supprime ensuite `data/`, `embeddings/`, `vector_db/` pour repartir de zéro.

## Réglages à confirmer avec le professeur (dans `config.py`)
- `N_TEST` : 100 ou 1000 (le PDF est incohérent)
- Validation disjointe de la KB (choix par défaut) ou incluse
- Liste d'images de test fournie ? -> `data/splits/test_ids_professor.json` (liste d'IDs val2017)

## Obligatoire / optionnel
- **Obligatoire** : étapes 0 à 3, 5 à 9. **Optionnel** : étape 4 (démo), détecteur d'objets (YOLO/DETR, non inclus), suppression des quasi-doublons visuels (`REMOVE_VISUAL_DUPLICATES`).

## Erreurs fréquentes
| Erreur | Solution |
|---|---|
| `MemoryError` à l'étape 1/2 (lecture de `instances_train2017.json`, 448 Mo) | fermer les autres programmes (4 Go de RAM libres suffisent), une seule fois |
| `Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` (le POS est ignoré sinon) |
| Téléchargement interrompu | relancer `00_download.py` (reprise automatique) |
| `Aucune image dans data/raw/train2017` | lancer l'étape 0 ou pointer `TRAIN_IMG_DIR` vers le dossier du prof |
| Étape 6 très lente | normal sur CPU ; relancer reprend où elle s'est arrêtée (`embeddings/_tmp`) |
| Changement de splits/modèle | supprimer `embeddings/_tmp` avant de relancer l'étape 6 |
