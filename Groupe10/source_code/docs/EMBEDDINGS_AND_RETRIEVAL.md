# Tâche de Yahia - Embeddings, Vector Database & Quality Control

Ce document explique **ce qui a été fait**, **pourquoi**, et **comment le
vérifier**. Le rapport de contrôle qualité généré se trouve dans
`reports/qc_report.md`.

## 1. Rappel de la tâche

> **Task:** Embeddings, Vector Database & Quality Control
>
> **What will he do?** Convert images and texts into embeddings, build a
> vector database and test similarity search.
>
> **What does he have?** Python and the required libraries and selected encoder.
>
> **What does he need from others?**
> - From Chahinez: Knowledge Base split IDs.
> - From Ramzi: processed images, captions and metadata.
>
> **What will he provide to others?** Image and text embeddings, a vector
> database and a quality-control report.

## 2. Choix techniques

| Choix | Justification |
|---|---|
| **CLIP ViT-B/32** | Un seul modèle encode images et textes dans le même espace (512-d), ce qui permet la recherche cross-modale texte→image et image→texte. Léger, exécutable sur CPU, déjà disponible dans l'environnement de l'équipe. |
| **Normalisation L2** | Le produit scalaire FAISS devient équivalent à la similarité cosinus, plus robuste aux variations de norme. |
| **FAISS `IndexFlatIP`** | Recherche exacte, suffisante pour 10 000 images / 50 000 légendes. Un mode `ivf` est disponible pour passer à l'échelle. |
| **Trois index** | `image_index` (texte→image), `caption_index` (image→texte), `joint_index` (recherche unifiée avec tag de modalité). |
| **JSONL + `.npy`** | Formats simples, lisibles, compatibles NumPy/Pandas, et ordre des lignes explicite. |
| **Téléchargement COCO** | Les images du dépôt sont des pointeurs Git LFS ; le pipeline récupère les vraies images via `coco_url`. |

## 3. Fichiers ajoutés

```
src/embeddings/
├── __init__.py
├── config.py            # chemins, constantes, modèle par défaut
├── data.py              # chargement des splits, validation, téléchargement COCO
├── encoder.py           # wrapper CLIP (images + textes), normalisation L2
├── build_embeddings.py  # CLI : calcule et sauvegarde les embeddings
├── build_vector_db.py   # CLI : construit les index FAISS
├── search.py            # CLI : recherche texte→image, image→texte, image→image
├── quality_control.py   # CLI : contrôles + rapport Markdown/JSON
└── pipeline.py          # CLI : tout le pipeline en une commande

requirements.txt
README.md
docs/YAHIA_TASK.md
```

## 4. Contrôles qualité effectués

### 4.1 Intégrité des embeddings

- Nombre de vecteurs = nombre de lignes de métadonnées.
- Dimension cohérente (512).
- Absence de `NaN` / `Inf`.
- Norme L2 proche de 1 (déviation maximale).
- Absence de vecteurs nuls.
- Doublons exacts (hash SHA-256 de chaque ligne).
- Quasi-doublons (cosinus ≥ 0,999 sur un échantillon borné).

### 4.2 Intégrité des métadonnées

- IDs d'images uniques.
- Chaque légende référence une image existante.
- Aucune image sans légende.
- Distribution du nombre de légendes par image.
- Légendes vides ou dupliquées.
- Présence réelle des fichiers images (et détection des pointeurs LFS).

### 4.3 Intégrité de la base vectorielle

- `ntotal` FAISS = nombre de métadonnées.
- Dimension de l'index = dimension des embeddings.
- Auto-récupération top-1 : chaque vecteur doit retrouver son propre index.
- Reconstruction exacte (index flat) : erreur absolue maximale.
- Taille et checksum SHA-256 des fichiers d'index.

### 4.4 Qualité de la recherche

Deux évaluations complémentaires :

1. **Split de test tenu à l'écart** (100 images, jamais utilisées pour la
   Knowledge Base) : on construit une galerie avec ces 100 images et on mesure
   si une légende retrouve sa propre image (texte→image) et si une image
   retrouve une de ses légendes (image→texte).
2. **Auto-récupération sur la Knowledge Base** : même mesure sur la galerie
   complète de la Knowledge Base, ce qui donne une estimation à l'échelle réelle.

Métriques : **Recall@1/5/10**, **MRR**, **rang médian**, **taux de succès**.

### 4.5 Performance

- Latence d'encodage d'une requête texte (CLIP, CPU).
- Latence de recherche FAISS top-5.
- Moyenne, p50, p95, min, max sur 20 requêtes.

## 5. Verdict automatique

| Verdict | Signification |
|---|---|
| `PASS` | Tous les contrôles passent. |
| `PASS WITH WARNINGS` | La base est utilisable, mais certaines métriques sont en dessous des seuils (ex. Recall@1 faible, doublons, normes légèrement déviantes). |
| `FAIL` | Problème bloquant : NaN/Inf, vecteurs nuls, incohérence de comptage/dimension, IDs dupliqués, images sans légende. |

Les seuils de Recall@1 sont configurables :

```powershell
python -m src.embeddings.quality_control `
  --min-recall1-heldout 0.30 `
  --min-recall1-kb 0.10
```

## 6. Limites

- CLIP est utilisé **zero-shot**, sans fine-tuning sur COCO : les scores sont
  une baseline.
- Sur une galerie de 100 images, le hasard donne déjà 1 % de Recall@1 ; les
  scores ne sont comparables qu'à taille de galerie égale.
- Les images sont téléchargées depuis COCO : le pipeline a besoin d'Internet
  ou d'images placées manuellement.
- L'index `flat` est exact mais coûteux en mémoire au-delà de quelques
  centaines de milliers de vecteurs ; utiliser `--index-type ivf` ensuite.

## 7. Commandes de vérification

```powershell
# 1. Démo rapide (500 images)
python -m src.embeddings.pipeline --kb-limit 500 --download

# 2. Vérifier les sorties
Get-ChildItem artifacts\embeddings
Get-ChildItem artifacts\vector_db
Get-Content reports\qc_report.md

# 3. Tester la recherche
python -m src.embeddings.search --mode text-to-image `
  --query "a man riding a motorcycle on a dirt road" --k 5

# 4. Exécution complète
python -m src.embeddings.pipeline --kb-limit 0 --download
```
