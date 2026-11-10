# Mini-projet ITI - Exploration et prétraitement des données

**A comparative study of RAG techniques for generating image captions**

Groupe : **Hamou Yasmine, Ouziala Ali et Lounis Aya**.

Ce dépôt traite uniquement la phase 1, prévue pour le **4 novembre 2026** et mise à jour le **7 novembre 2026** selon les précisions transmises. Il prépare les données et vérifie la recherche d'images proches. Il ne génère pas de captions finales et n'entraîne pas de RAG.

**Contenu de GitHub :** code, [description du jeu de données](DATASET.md), listes d'IDs, statistiques, figures, résultats de validation retrieval et rapports PDF/Markdown. Les photos, les poids CLIP, les matrices et les index volumineux restent locaux et sont exclus par `.gitignore`.

## Tester maintenant sur ce PC

Dans VS Code, ouvrir ce dossier, puis **Terminal > New Terminal** (PowerShell) :

```powershell
.\imp\Scripts\python.exe -m src.cli --offline retrieve --image-id 490 --top-k 5 --show
```

La galerie montre l'image 490 et cinq images proches de la base, avec leurs scores et leurs captions COCO existantes. Les résultats sont enregistrés dans `reports/retrieval/manual_490_k5/` : `gallery.html`, `nearest_images.png`, `results.json` et `results.csv`.

Pour dix voisins :

```powershell
.\imp\Scripts\python.exe -m src.cli --offline retrieve --image-id 133 --top-k 10 --show
```

L'image 133 appartient à la base : elle est exclue de ses propres résultats. Les IDs COCO ne sont pas des numéros de ligne. L'ID 133 correspond à `000000000133.jpg`. Pour choisir visuellement une image, parcourir `data/raw/train2017/` en affichage grandes icônes.

Pour une photo personnelle, la copier dans `examples/ma_photo.jpg`, puis :

```powershell
.\imp\Scripts\python.exe -m src.cli --offline retrieve --image-path examples/ma_photo.jpg --top-k 5 --show
```

`TESTER.cmd` lance aussi la démonstration de l'image 490 lorsque l'environnement `imp` est installé et les données restaurées.

## Données et exigences de la phase 1

| Usage | Source | Effectif | Règle |
|---|---|---:|---|
| Base de connaissances | COCO train2017 | 10 000 | Seule source des index de recherche |
| Validation | COCO train2017 | 100 | IDs différents de ceux de la base |
| Test étendu | COCO val2017 | 1 000 | Références séparées de la base |
| Ancien test conservé | Sous-ensemble du test étendu | 100 | Comparaison descriptive, sans double comptage |

**Total : 11 100 images distinctes**, dont 10 100 de train2017. Les 100 anciennes images de test sont incluses dans les 1 000. Leurs IDs restent dans `artifacts/splits/comparison/test_100_ids.json` et leurs statistiques dans `reports/eda.json`, vue `test_100`.

Les 900 IDs ajoutés ont été tirés dans les images val2017 restantes, avec une stratification par nombre d'annotations et la graine 42. Les IDs de la base et de la validation n'ont pas changé. L'ancienne version et ses preuves sont conservées dans `reports/releases/phase1_test100_before_extension/`.

**Limite externe :** aucune liste d'IDs commune à toute la classe n'a été fournie. Le test est donc un échantillon local gelé, pas une liste commune officiellement approuvée. L'enseignante doit valider la préparation.

## Méthode retenue

1. Télécharger les images sélectionnées et les quatre JSON d'annotations officiels COCO.
2. Auditer les fichiers : présence, décodage, dimensions, cinq captions par image, doublons exacts et absence de recouvrement entre base, validation et test.
3. Préparer les images en mémoire : correction EXIF, RGB, redimensionnement et recadrage central 224 x 224 avec le processeur officiel CLIP, puis normalisation.
4. Conserver `caption_raw` inchangé. Créer `caption_clean` en normalisant uniquement les espaces ; casse et ponctuation conservées.
5. Associer les objets aux images avec les annotations COCO (`image_id`, catégorie, boîte). Les couleurs, formes et contextes issus des captions restent des heuristiques identifiées.
6. Enregistrer `source=coco_human_reference`, `confidence=1` et le seuil de conservation des références humaines à 1. Cette convention ne s'applique pas aux attributs déduits et n'est pas un score de détection.
7. Produire les statistiques et figures, notamment avant/après et pour les tests 100/1000.
8. Utiliser CLIP préentraîné pour obtenir 10 000 vecteurs image et 50 000 vecteurs texte de dimension 512, normalisés en float32. Les vecteurs existants ont été réutilisés après vérification que leurs véritables entrées n'ont pas changé.
9. Construire les deux index FAISS `IndexFlatIP`. Le produit scalaire des vecteurs normalisés donne la similarité cosinus.
10. Vérifier les correspondances et lancer les 100 requêtes de validation réelles. Il s'agit d'un contrôle fonctionnel, sans métriques finales de génération.

La détection automatique dans les pixels est une recommandation facultative ; aucun YOLO/DETR n'a été exécuté. Les statistiques d'objets portent sur les annotations COCO, pas sur des prédictions annoncées comme réelles.

## Rapport, statistiques et preuves

- [Rapport anglais (PDF)](output/pdf/Phase1_report_EN.pdf) et [version Markdown](output/pdf/Phase1_report_EN.md).
- [Jeu de données et règles de partage](DATASET.md).
- [Figures et validation retrieval](results/README.md).
- [Vérification des exigences](reports/phase1_review_20261107.md), [audit des données](reports/audit.json), [contrôle qualité](reports/quality_control.json), [statut](reports/run_status.json).
- `reports/eda.json`, `reports/figures/` et `reports/tables/` : effectifs, classes, complexité, captions, vocabulaire, POS, relations, cooccurrences et diversité.
- `reports/phase1_upgrade.json` : preuve que les IDs base/validation et les matrices CLIP ont été conservés.
- `reports/unit_tests_latest.json` : tests du logiciel sur petites fixtures, distincts des résultats COCO réels.

Les rapports sont fondés sur les fichiers réellement produits. Les captions affichées par la recherche sont des références humaines récupérées, pas des captions générées.

**Pour découvrir le projet sans le connaître :** commencer par le pipeline, puis lire les partitions et les choix de prétraitement. Le cas de l'image COCO 490 explique les réussites et les erreurs de retrieval. Les étapes de reproduction figurent à la fin du rapport. Les figures et le résultat de validation le plus récent sont regroupés dans [`results/`](results/). Une *caption* est une phrase décrivant une photo ; un *embedding* est sa représentation par une liste de nombres ; FAISS compare ces listes pour retrouver les photos proches.

L'exemple détaillé est vérifié dans [pedagogical_example.json](reports/pedagogical_example.json) : le vecteur de requête réellement sauvegardé est recherché à nouveau dans FAISS et les cinq scores sont recalculés par produit scalaire. Les deux cas commentés sont des illustrations, pas une mesure du taux de réussite. La similarité ne garantit pas que les objets ou les actions correspondent.

L'annotation COCO 918 de l'image 200365 comporte une boîte de hauteur nulle. Elle est conservée pour les comptes de catégories et marquée `bbox_valid=false`, puis exclue des seules relations géométriques et illustrations de boîtes. Les originaux restent inchangés. Les chaînes de captions partagées entre images distinctes sont signalées sans suppression automatique. Les comptes incluent les annotations de foule ; ce ne sont pas des recensements exacts des objets physiques.

## Reproduire depuis le code seul sur un nouveau PC Windows

Installer Python 3.12 x64, ouvrir le dossier dans VS Code et exécuter dans l'ordre, en s'arrêtant à la première erreur :

```powershell
py -3.12 -m venv imp
.\imp\Scripts\python.exe -m pip install -r requirements.txt
$impPython = ".\imp\Scripts\python.exe"
& $impPython -m src.cli download-annotations
& $impPython -m src.cli download-images
& $impPython -m src.cli download-nlp
& $impPython -m src.cli download-model
& $impPython -m src.cli metadata
& $impPython -m src.cli eda
& $impPython -m src.cli --offline embeddings
& $impPython -m src.cli index
& $impPython -m src.cli --offline validate-retrieval --top-k 5
& $impPython -m src.cli --offline qc
& $impPython scripts/review_phase1.py
& $impPython -m src.cli report
```

Réutiliser les IDs fournis dans `artifacts/splits/`. **Ne pas lancer `splits --force`** pour reproduire le projet. Le calcul des embeddings prend du temps sur CPU ; il est évité si les matrices préparées sont restaurées. Les versions effectivement utilisées sont enregistrées dans les manifestes. L'exécution locale utilise Python 3.12.10 et un environnement de développement partageant des bibliothèques préinstallées : une installation propre sur un autre PC n'a pas été vérifiée.

## Dossier préparé pour Google Drive

La version actualisée est **`releases/Projet_ITI_Drive_v2/`**. Le code, les rapports et les statistiques sont consultables directement. Les gros fichiers sont répartis entre :

- `data/01_images_annotations.zip` ;
- `models/02_clip_nltk.zip` ;
- `artifacts/03_metadata_embeddings_faiss.zip`.

Importer **ce dossier entier** dans Drive : Nouveau > Importer un dossier. Le destinataire suit son `README.md`. Après téléchargement, `py -3.12 restaurer_donnees.py` vérifie les archives et remet les fichiers aux bons chemins. Python et les bibliothèques doivent être installés une fois ; les images, le modèle, les embeddings et les index sont déjà préparés.

Les anciennes distributions `Projet_ITI_Drive` et le ZIP autonome Windows décrivent la version antérieure à 100 images de test. Elles sont conservées comme anciennes versions et ne doivent pas être présentées comme la version actualisée.

## Reconstruire le rapport généré par le code et les archives

```powershell
py -3.12 -m pip install --target tmp/report_deps reportlab pypdf
.\imp\Scripts\python.exe scripts/build_phase1_report.py
.\imp\Scripts\python.exe scripts/build_drive_package.py
```

Attention : `build_phase1_report.py` regénère les rapports à partir des données locales et remplace les fichiers du même nom dans `output/pdf/`. Le PDF anglais et son édition Markdown actuellement publiés sont ceux du rapport fourni ; sauvegardez-les avant de lancer ce générateur. Le générateur d'archives refuse d'écraser un dossier de distribution existant. L'utilitaire `extend-test` a servi uniquement à la migration documentée du test initial ; il n'est pas nécessaire pour un nouveau téléchargement utilisant les listes déjà gelées.

Référence principale : [cahier des charges](docs/project/Image%20processing%20mini-project%20english%20version.pdf). Les cours et le TP sont du contexte. [Précisions transmises](docs/project/teacher_clarifications_20261107.md). Les exemples du TP ne sont pas les données scientifiques COCO.
