r"""Build the French reading report and English phase-1 report from real artifacts.

Reporting-only dependencies: reportlab, pypdf (not required by the retrieval CLI).
Example: py -3.12 -m pip install --target tmp/report_deps reportlab pypdf
Then: .\imp\Scripts\python.exe scripts/build_phase1_report.py
"""
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tmp/report_deps")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image, Preformatted)

from src.preprocessing.metadata import load_rgb
from src.utils.core import file_hash, load_config, read_json, write_json

OUT = ROOT / "output/pdf"
FIG = ROOT / "reports/figures"
REVIEW = ROOT / "reports/phase1_review_20261107.json"


def make_figures():
    """Run the actual cached CLIP image processor, without model inference."""
    from transformers import AutoProcessor
    cfg = load_config()
    proc = AutoProcessor.from_pretrained(
        cfg["embedding"]["model"], revision=cfg["embedding"]["revision"],
        cache_dir=str(ROOT / "data/interim/cache/huggingface"),
        local_files_only=True, trust_remote_code=False)
    original_path = ROOT / "data/raw/train2017/000000000133.jpg"
    original_hash = file_hash(original_path)
    rgb = load_rgb(original_path)
    pixels = proc(images=[rgb], return_tensors="np")["pixel_values"]
    assert pixels.shape == (1, 3, 224, 224) and np.isfinite(pixels).all()
    ip = proc.image_processor
    mean, std = np.array(ip.image_mean), np.array(ip.image_std)
    visible = np.clip(pixels[0].transpose(1, 2, 0) * std + mean, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb); axes[0].set_title(f"Original RGB: {rgb.width} x {rgb.height}")
    axes[1].imshow(visible); axes[1].set_title("CLIP input: 224 x 224 (display view)")
    for ax in axes: ax.axis("off")
    fig.suptitle("COCO image ID 133 - actual cached processor, no model inference")
    fig.tight_layout(); fig.savefig(FIG / "phase1_before_after_clip.png", dpi=170); plt.close(fig)
    rgb.close()
    assert original_hash == file_hash(original_path)
    first = json.loads((ROOT / "artifacts/metadata/knowledge_base.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["image_id"] == 133
    objects = [o for o in first["objects"] if o["bbox_valid"]][:3]
    fig, axes = plt.subplots(1, len(objects) + 1, figsize=(10, 3.1))
    with PILImage.open(original_path) as raw:
        axes[0].imshow(raw); axes[0].set_title("Image 133 / COCO boxes")
        from matplotlib.patches import Rectangle
        for o, ax in zip(objects, axes[1:]):
            x, y, w, h = o["bbox"]
            axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.3))
            ax.imshow(raw.crop((x, y, x + w, y + h)))
            ax.set_title(f"{o['category']}\nannotation {o['id']}", fontsize=9)
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "phase1_coco_object_regions.png", dpi=170); plt.close(fig)
    write_json(ROOT / "reports/phase1_processor_example.json", {
        "image_id": 133, "input_sha256_before": original_hash,
        "input_sha256_after": file_hash(original_path),
        "processor_revision": cfg["embedding"]["revision"],
        "tensor_shape": list(pixels.shape), "dtype": str(pixels.dtype),
        "finite": bool(np.isfinite(pixels).all()),
        "mean": mean.tolist(), "std": std.tolist(),
        "display_policy": "Denormalized only for the illustration; actual encoder receives normalized tensor",
        "object_regions_source": "Crops from existing human COCO boxes; no detector run",
        "region_annotation_ids": [o["id"] for o in objects],
    })
    from src.preprocessing.metadata import load_metadata
    from src.exploration.eda import word_tokens
    test = load_metadata(cfg, "test")
    legacy = set(read_json(ROOT / "artifacts/splits/comparison/test_100_ids.json"))
    first100 = [r for r in test if r["image_id"] in legacy]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, label, extractor in [(axes[0], "Object annotations / image", lambda rows: [r["object_count"] for r in rows]),
                                  (axes[1], "Words / original caption", lambda rows: [len(word_tokens(c["caption_raw"])) for r in rows for c in r["captions"]])]:
        full, small = np.array(extractor(test)), np.array(extractor(first100))
        bins = np.arange(0, max(full.max(), small.max()) + 2) - .5
        ax.hist(full, bins=bins, weights=np.ones(len(full))/len(full), alpha=.6, label="Test 1000")
        ax.hist(small, bins=bins, weights=np.ones(len(small))/len(small), histtype="step", linewidth=1.5, label="Original test 100")
        ax.set(xlabel=label, ylabel="Relative frequency")
        ax.legend(fontsize=8)
    fig.suptitle("Descriptive comparison - original 100 nested in extended 1000")
    fig.tight_layout(); fig.savefig(FIG / "phase1_test100_test1000.png", dpi=170); plt.close(fig)


class Book:
    def __init__(self, lang):
        self.lang = lang
        self.pages = []
        self.current = None

    def tr(self, fr, en): return fr if self.lang == "fr" else en

    def page(self, fr, en):
        self.current = {"title": self.tr(fr, en), "blocks": []}
        self.pages.append(self.current)

    def p(self, fr, en=None): self.current["blocks"].append(("p", self.tr(fr, en or fr)))
    def h(self, fr, en): self.current["blocks"].append(("h", self.tr(fr, en)))
    def code(self, text): self.current["blocks"].append(("code", text))
    def table(self, headers, rows): self.current["blocks"].append(("table", (headers, rows)))
    def figure(self, name, height, fr, en): self.current["blocks"].append(("figure", (name, height, self.tr(fr, en))))


def content(lang):
    b = Book(lang); t = b.tr
    eda = read_json(ROOT / "reports/eda.json")["splits"]
    review = read_json(REVIEW); ba = review["before_after"]
    kb = eda["knowledge_base"]
    retrieval = read_json(ROOT / "reports/retrieval/validation/results.json")
    em = read_json(ROOT / "artifacts/embeddings/manifest.json")
    indexes = read_json(ROOT / "artifacts/indexes/manifest.json")["indexes"]
    b.page("Exploration et prétraitement des données", "Data exploration and pre-processing")
    b.p("Université d'Alger 1 Benyoucef Benkhedda\nMaster 1 - Ingénierie des systèmes informatiques intelligents\nModule : Introduction au Traitement d'Images\nAnnée universitaire 2026-2027",
        "University of Algiers 1 Benyoucef Benkhedda\nMaster 1 - Intelligent Computer Systems Engineering\nModule: Introduction to Image Processing\nAcademic year 2026-2027")
    b.h("Mini-projet", "Mini-project")
    b.p("A comparative study of RAG techniques for generating image captions")
    b.h("Rapport de phase 1", "Phase-1 report")
    b.h("Réalisé par", "Prepared by")
    b.p("Hamou Yasmine\nOuziala Ali\nLounis Aya")
    b.p("Ce rapport explique comment les images COCO ont été sélectionnées, vérifiées et préparées, puis comment leurs représentations CLIP ont été organisées dans FAISS. Il décrit uniquement le travail réellement exécuté. Un lecteur peut suivre les commandes de reproduction à la fin du document.",
        "This report explains how COCO images were selected, checked and prepared, and how their CLIP representations were organised in FAISS. It describes only work that was actually executed. The reproduction commands at the end allow the reader to follow the same steps.")
    b.table([t("Repère", "Milestone"), t("Date", "Date")], [
        [t("Échéance de cette phase", "Phase deadline"), "4 November 2026"],
        [t("Exécution principale des données", "Main data preparation run"), "28 October 2026"],
        [t("Revue et rédaction mises à jour", "Review and report update"), "7 November 2026"]])
    b.p("État de la version décrite : 10 000 images pour la base, 100 pour la validation et un premier test de 100 images. La cible de 1 000 images de test, précisée ensuite par l'enseignante, reste à préparer. Ce rapport ne présente donc pas la phase comme entièrement conforme à cette nouvelle cible.",
        "State of the reviewed release: 10,000 knowledge-base images, 100 validation images and an initial 100-image test set. The 1,000-image test target clarified by the teacher still needs to be prepared. This report therefore does not claim full compliance with the updated target.")
    b.p("Aucune génération finale de captions, aucun entraînement de RAG et aucune évaluation finale ne sont inclus.",
        "Final caption generation, RAG training and final evaluation are outside this report.")

    b.page("1. Objectif et guide de lecture", "1. Purpose and reading guide")
    b.p("Une caption est une phrase qui décrit une image. Plus tard, le RAG cherchera des exemples proches pour aider un modèle à écrire cette phrase. Avant cela, il faut disposer de données fiables : une image lisible, ses cinq phrases de référence, ses informations associées et une représentation numérique utilisable pour la recherche.",
        "A caption is a sentence describing an image. Later, RAG will retrieve similar examples to help a model write that sentence. First, reliable data are needed: a readable image, its five reference sentences, associated information and a numerical representation that can be searched.")
    b.p("Le mot entraînement désigne ici la partie train2017 utilisée pour constituer la base. CLIP est utilisé avec ses poids préentraînés : nous ne l'avons pas entraîné ni ajusté sur ces images.",
        "Training refers here to the train2017 source used to build the knowledge base. CLIP is used with pretrained weights; it was neither trained nor fine-tuned on this subset.")
    b.table([t("Étape", "Step"), t("Question traitée", "Question answered"), t("Page", "Page")], [
        [t("Données et séparation", "Data and splitting"), t("Quelles images ? Pour quel usage ?", "Which images, for which purpose?"), "3"],
        [t("Audit", "Audit"), t("Peut-on utiliser les fichiers ?", "Are the files usable?"), "4"],
        [t("Métadonnées", "Metadata"), t("D'où viennent les objets et attributs ?", "Where do objects and attributes come from?"), "5"],
        [t("Prétraitement", "Pre-processing"), t("Que change-t-on dans l'image et le texte ?", "What changes in images and text?"), "6-7"],
        [t("Exploration", "Exploration"), t("Que contient réellement ce sous-ensemble ?", "What does this subset actually contain?"), "8-11"],
        ["CLIP / FAISS", t("Comment chercher des images proches ?", "How are similar images retrieved?"), "12-13"],
        [t("Reproduction", "Reproduction"), t("Quelles commandes relancer ?", "Which commands should be run?"), "14-15"],
        [t("Bilan et limites", "Status and limitations"), t("Que reste-t-il à compléter ?", "What still needs to be done?"), "16"]])
    b.h("Comment lire les résultats", "How to read the results")
    b.p("Les nombres proviennent des JSON et CSV du projet. Les figures proviennent des images COCO et des statistiques sauvegardées. Chaque étape donne ses fichiers d'entrée, sa méthode et sa sortie. Les termes techniques sont définis au moment où ils apparaissent.",
        "Numbers come from the project's JSON and CSV files. Figures use COCO images and saved statistics. Each step explains its input, method and output. Technical terms are defined when first used.")
    b.p("Le code est organisé en modules correspondant aux étapes. La commande python -m src.cli permet de les lancer dans le terminal. Les notebooks appellent ces mêmes modules : ils ne constituent pas une seconde méthode de calcul. Les fichiers de sortie permettent de vérifier les résultats sans refaire tout le calcul.",
        "The code is organised into modules matching these steps. python -m src.cli runs them from the terminal. Notebooks call the same modules rather than implementing a second computation method. Output files allow results to be checked without repeating the entire computation.")

    b.page("2. Données COCO et séparation des jeux", "2. COCO data and split separation")
    b.p("COCO fournit les photographies et des fichiers JSON. Les fichiers captions associent des phrases aux images. Les fichiers instances associent des catégories d'objets et leurs boîtes englobantes aux images. Le nom du dossier officiel est val2017, et non val17.",
        "COCO provides photographs and JSON annotation files. Caption files link sentences to images. Instance files link object categories and bounding boxes to images. The official directory name is val2017, not val17.")
    b.table([t("Usage", "Use"), "Source", t("Présent", "Present"), t("Cible", "Target")], [
        [t("Base de connaissances", "Knowledge base"), "train2017", "10 000", "10 000"],
        [t("Validation séparée", "Separate validation"), "train2017", "100", "100"],
        [t("Premier test conservé", "Preserved initial test"), "val2017", "100", "100"],
        [t("Test étendu", "Extended test"), "val2017", t("Non préparé", "Not prepared"), "1 000"]])
    b.p("10 000 + 100 = 10 100 images distinctes de train2017. La validation ne fait pas partie des 10 000. Avec les 100 images du premier test, la version actuelle contient 10 200 images sélectionnées au total.",
        "10,000 + 100 = 10,100 distinct train2017 images. Validation is outside the 10,000-image base. Including the initial 100-image test, the current release contains 10,200 selected images in total.")
    b.h("Pourquoi séparer les usages ?", "Why separate these uses?")
    b.p("La base contient les exemples que FAISS peut retrouver. La validation sert aux vérifications et, plus tard, aux choix de paramètres. Le test doit rester réservé à l'évaluation. Ses phrases de référence ne doivent pas servir à construire une requête ni être ajoutées à la base.",
        "The knowledge base contains examples FAISS may retrieve. Validation supports checks and later parameter choices. Test data must remain reserved for evaluation. Their reference captions must not construct a query or enter the knowledge base.")
    b.h("Comment les images ont été choisies", "How images were selected")
    b.p("Les candidates doivent avoir leurs informations d'instances et exactement cinq captions. Les exclusions d'éligibilité sont enregistrées : 315 IDs de train2017 et 13 de val2017. Ce n'est pas un décompte d'images corrompues. Les scènes sont réparties selon le nombre d'annotations : simples (0 à 3), moyennes (4 à 7), complexes (8 ou plus). Les quotas sont proportionnels ; les arrondis sont distribués par la méthode des plus grands restes. Le tirage utilise la graine 42.",
        "Candidates must have matching instance-image information and exactly five captions. Eligibility exclusions are recorded: 315 train2017 IDs and 13 val2017 IDs. These are not corrupt-image counts. Scenes are grouped by annotation count: simple (0-3), medium (4-7), complex (8 or more). Quotas are proportional, with largest-remainder rounding. Sampling uses seed 42.")
    b.p("Les listes sont gelées dans artifacts/splits/*_ids.json et contrôlées par empreinte SHA256. Le test actuel reste provisoire : aucune liste commune fournie par l'enseignante n'est présente. Changer seulement test_size puis relancer splits --force peut aussi modifier les autres listes ; ce n'est pas la procédure proposée pour ajouter le test de 1 000 images.",
        "Lists are frozen in artifacts/splits/*_ids.json and checked using a SHA256 fingerprint. The current test remains provisional: no common teacher-supplied ID list is present. Merely changing test_size and rerunning splits --force can also change the other lists; this is not the proposed way to add the 1,000-image test.")

    b.page("3. Téléchargement et audit des données", "3. Downloading and auditing the data")
    b.p("Le téléchargement utilise le stockage officiel COCO sur S3, en HTTPS. Seules les photographies des IDs gelés sont téléchargées. L'archive d'annotations est décompressée pour obtenir captions_train2017.json, captions_val2017.json, instances_train2017.json et instances_val2017.json. Les originaux restent dans data/raw/.",
        "Downloads use official COCO S3 storage over HTTPS. Only photographs listed in the frozen splits are downloaded. The annotation archive provides captions_train2017.json, captions_val2017.json, instances_train2017.json and instances_val2017.json. Originals stay in data/raw/.")
    b.p("Avant toute préparation, le programme vérifie l'existence de chaque fichier, le décodage complet avec Pillow, ses dimensions, ses cinq captions et les identifiants associés. Il calcule des empreintes des fichiers et des pixels RGB pour rechercher des copies exactes. Il vérifie aussi les intersections entre listes d'images et entre IDs de captions.",
        "Before preparation, the program checks each file's existence, full Pillow decoding, dimensions, five captions and associated IDs. File and RGB-pixel fingerprints identify exact copies. It also checks intersections between image lists and between caption IDs.")
    b.table([t("Contrôle", "Check"), t("Résultat observé", "Observed result")], [
        [t("Images attendues / présentes / lisibles", "Expected / present / readable images"), "10 200 / 10 200 / 10 200"],
        [t("Images manquantes ou illisibles", "Missing or unreadable images"), "0"],
        [t("Captions par image", "Captions per image"), "5"],
        [t("IDs d'images communs entre jeux", "Shared image IDs between splits"), "0"],
        [t("Copies exactes de fichiers ou pixels", "Exact file or pixel copies"), "0"],
        [t("Boîte dégénérée documentée", "Documented degenerate box"), "1"],
        [t("Textes normalisés partagés entre jeux", "Shared normalised caption strings"), "19"]])
    b.h("Une anomalie conservée et expliquée", "One anomaly retained and explained")
    b.p("L'annotation 918 de l'image 200365 contient la boîte [296.65, 388.33, 1.03, 0.0] : sa hauteur est nulle. L'audit initial s'est arrêté. La décision a été de garder l'image, les phrases et la catégorie d'objet, mais d'indiquer bbox_valid=false. Cette boîte n'entre ni dans les calculs géométriques ni dans le dessin des rectangles. Les fichiers COCO n'ont pas été corrigés manuellement.",
        "Annotation 918 of image 200365 contains box [296.65, 388.33, 1.03, 0.0], with zero height. The initial audit stopped. The image, captions and object category were retained, with bbox_valid=false. This box is excluded from geometry and rectangle drawing. Official COCO files were not manually changed.")
    b.p("Les 19 phrases identiques concernent des images et des annotations distinctes. Une phrase courante peut décrire plusieurs photos ; ce signal est documenté sans supprimer ces images. La recherche de quasi-doublons visuels reste une limite. Preuves : reports/audit.json, reports/tables/audit_issues.csv, shared_caption_strings.csv et l'audit initial archivé.",
        "The 19 identical strings belong to distinct images and annotations. A common phrase may describe several photographs; this warning is documented without deleting those images. Perceptual near-duplicate detection remains a limitation. Evidence: reports/audit.json, reports/tables/audit_issues.csv, shared_caption_strings.csv and the archived initial audit.")

    b.page("4. Relier les images à leurs métadonnées", "4. Linking images to their metadata")
    b.p("Une métadonnée est une information associée à l'image. Ici, chaque ligne JSONL correspond à une image : image_id, nom du fichier, jeu, dimensions, empreinte, cinq captions, objets, catégories et boîtes. JSONL signifie simplement un objet JSON par ligne. Cela permet de retrouver une image sans perdre les informations qui lui appartiennent.",
        "Metadata are information associated with an image. Each JSONL row represents one image: image_id, filename, split, dimensions, fingerprint, five captions, objects, categories and boxes. JSONL simply means one JSON object per line. This keeps an image and its information linked.")
    b.table([t("Information", "Information"), t("Source réelle", "Actual source"), t("Interprétation", "Interpretation")], [
        [t("Objets et boîtes", "Objects and boxes"), "coco_annotation", t("Annotations humaines COCO", "Human COCO annotations")],
        [t("Couleurs, formes, contexte", "Colours, shapes, context"), "caption_heuristic", t("Mots repérés dans les captions", "Words found in captions")],
        [t("Relations spatiales", "Spatial relations"), "bbox_heuristic", t("Calcul géométrique sur les boîtes", "Geometry computed from boxes")]])
    b.figure("phase1_coco_object_regions.png", 155,
             "Exemple réel : régions découpées à partir des boîtes COCO de l'image 133. Ce sont des annotations existantes, pas les sorties d'un détecteur lancé sur les pixels.",
             "Actual example: regions cropped using image 133's COCO boxes. These are existing annotations, not outputs of a detector run on pixels.")
    b.h("Détection visuelle et confiance : ce qui a été fait", "Visual detection and confidence: actual implementation")
    b.p("Aucun YOLO, DETR ou autre détecteur n'a été exécuté. L'extraction visuelle recommandée par l'enseignante reste donc une amélioration possible. CLIP fournit un vecteur global ; il ne fournit pas ici une liste d'objets avec des boîtes et des scores. Les statistiques d'objets actuelles portent sur les annotations COCO.",
        "No YOLO, DETR or other detector was run. The teacher's recommended visual extraction remains a possible improvement. CLIP provides a global vector here, not a list of objects with boxes and confidence scores. Current object statistics describe COCO annotations.")
    b.p("La valeur 1 mentionnée pour les captions humaines doit être présentée comme une convention sur ces références. Elle ne donne pas une certitude aux attributs extraits automatiquement. Le schéma actuel ne contient pas de champ confidence. Il n'applique donc pas encore explicitement cette convention. Aucun seuil de détecteur n'est à déclarer pour la version actuelle.",
        "The value 1 mentioned for human captions should be presented as a reference-source convention. It does not make automatically extracted attributes certain. The current schema has no confidence field and does not yet explicitly encode that convention. No detector threshold applies to the current release.")

    b.page("5. Prétraitement des images", "5. Image pre-processing")
    b.p("Le prétraitement prépare le format attendu par CLIP. On ouvre le JPEG, on corrige son orientation EXIF si nécessaire, puis on le convertit en RGB. RGB conserve les trois canaux de couleur attendus par l'encodeur. Nous gardons cette information de couleur pour représenter le contenu des photographies.",
        "Pre-processing prepares CLIP's expected input. The JPEG is opened, EXIF orientation is corrected when needed, and the image is converted to RGB. RGB retains the three colour channels expected by the encoder. Colour information is preserved when representing photograph content.")
    b.p("Le processeur officiel du modèle redimensionne le petit côté à 224 pixels en conservant le rapport largeur/hauteur, puis réalise un recadrage central de 224 x 224. Il met les valeurs à l'échelle puis normalise chaque canal selon les paramètres du modèle. Ce recadrage peut retirer des objets en bordure : c'est une limite à garder en tête.",
        "The model's official processor resizes the shorter side to 224 pixels while preserving aspect ratio, then takes a central 224 x 224 crop. It rescales values and normalises each channel using the model parameters. Cropping may remove peripheral objects; this is a limitation.")
    b.figure("phase1_before_after_clip.png", 220,
             "Image 133 avant et après le processeur réel mis en cache. L'aperçu de droite est dénormalisé uniquement pour l'affichage. Le tenseur transmis à CLIP reste normalisé.",
             "Image 133 before and after the actual cached processor. The right preview is denormalised only for display; CLIP receives the normalised tensor.")
    b.code("mean = [0.48145466, 0.45782750, 0.40821073]\nstd  = [0.26862954, 0.26130258, 0.27577711]\nnormalised_channel = (channel / 255 - mean) / std")
    b.p("Le contrôle illustré a produit un tenseur fini de forme (1, 3, 224, 224). L'empreinte du JPEG est identique avant et après. Les images transformées sont utilisées en mémoire ; aucun second lot de JPEG n'est écrit dans data/processed/. L'exemple et ses paramètres sont enregistrés dans reports/phase1_processor_example.json.",
        "The illustrated check produced a finite tensor of shape (1, 3, 224, 224). The JPEG fingerprint is unchanged. Transformed images are used in memory; no second JPEG collection is written into data/processed/. The example and its parameters are recorded in reports/phase1_processor_example.json.")

    b.page("6. Captions et comparaison avant/après", "6. Captions and before/after comparison")
    b.p("Les captions originales sont les références humaines. Le champ caption_raw conserve leur texte exact, y compris les espaces. Un champ séparé, caption_clean, sert à l'indexation : les espaces multiples et ceux en début ou fin de phrase sont normalisés. La casse et la ponctuation sont conservées avec la configuration actuelle. Il n'y a ni traduction, ni réécriture, ni suppression des mots fréquents.",
        "Original captions are human references. caption_raw preserves their exact text, including spaces. A separate caption_clean field is used for indexing: repeated, leading and trailing whitespace is normalised. Case and punctuation are preserved in the current configuration. There is no translation, rewriting or stop-word removal.")
    b.table([t("Jeu", "Split"), t("Images avant/après", "Images before/after"), t("Captions", "Captions"), t("Champs nettoyés modifiés", "Changed clean fields")], [
        [s, f"{v['images_before']} / {v['images_after']}", str(v['captions_after']), str(v['clean_caption_fields_different_from_raw'])] for s, v in ba.items()])
    b.p(f"Dans la base, {ba['knowledge_base']['clean_caption_fields_different_from_raw']} captions sur 50 000 ont un champ nettoyé différent du texte brut. Le nombre de caractères passe de {ba['knowledge_base']['raw_characters']} à {ba['knowledge_base']['clean_characters']}. Les 50 000 références brutes restent inchangées. Ces valeurs ont été recomptées sur les métadonnées réelles lors de la revue.",
        f"In the knowledge base, {ba['knowledge_base']['clean_caption_fields_different_from_raw']} of 50,000 captions have a clean field different from the raw text. Character count changes from {ba['knowledge_base']['raw_characters']} to {ba['knowledge_base']['clean_characters']}. All 50,000 raw references remain unchanged. These values were recounted from actual metadata during the review.")
    example = ba["knowledge_base"]["cleaning_example"]
    b.code("caption_id: " + str(example["caption_id"]) + "\nraw:   " + repr(example["caption_raw"]) + "\nclean: " + repr(example["caption_clean"]))
    b.h("Ce que compare cette exploration", "What this comparison measures")
    b.p("Avant/après désigne ici la conservation des effectifs, le nettoyage du texte, les formats d'image et l'exclusion géométrique de la boîte invalide. Les 72 507 annotations d'objets de la base restent présentes ; 72 506 boîtes sont utilisables pour les relations. La distribution des catégories ne change donc pas. Nous ne prétendons pas avoir amélioré un score de génération.",
        "Before/after refers here to retained counts, text cleaning, image formats and exclusion of the invalid box from geometry. All 72,507 knowledge-base object annotations are retained; 72,506 boxes are usable for relations. Category frequencies therefore remain unchanged. No generation-score improvement is claimed.")
    b.p("Les statistiques linguistiques des pages suivantes utilisent le texte brut. Le tokenizer NLTK Treebank découpe les phrases ; seuls les tokens contenant une lettre sont comptés comme mots. Ils passent en minuscules pour le vocabulaire. Pour les catégories grammaticales (POS), le modèle reçoit la séquence complète avec sa casse d'origine. Une mesure en mots n'est pas le nombre de sous-tokens utilisé par CLIP.",
        "The following language statistics use raw text. NLTK's Treebank tokenizer splits sentences; only tokens containing a letter count as words. They are lowercased for vocabulary counting. For part-of-speech (POS) tagging, the model receives the full original-case sequence. Word counts are not CLIP sub-token counts.")

    b.page("7. Exploration : fréquence des objets", "7. Exploration: object frequencies")
    b.p("La base contient 72 507 annotations d'objets. Le graphique compte les occurrences d'annotations, pas le nombre d'images contenant chaque catégorie. Une image avec plusieurs personnes contribue plusieurs fois à person. Les annotations de foule restent comptées comme des entrées d'annotation.",
        "The knowledge base contains 72,507 object annotations. The chart counts annotation instances, not images containing each category. An image with several people contributes several entries to person. Crowd annotations remain counted as annotation entries.")
    b.figure("knowledge_base_top50_objects.png", 485,
             "Les 50 catégories les plus fréquentes dans les 10 000 images. Source : reports/tables/knowledge_base_object_classes.csv.",
             "The 50 most frequent categories across 10,000 images. Source: reports/tables/knowledge_base_object_classes.csv.")
    b.p("La catégorie person domine avec 22 670 annotations, devant car (3 815) et chair (3 151). Ce déséquilibre décrit notre sous-ensemble. Il ne permet pas, à lui seul, de conclure à la qualité d'un futur modèle.",
        "person leads with 22,670 annotations, followed by car (3,815) and chair (3,151). This imbalance describes the selected subset; it does not by itself establish a future model's quality.")

    b.page("8. Exploration : complexité des scènes", "8. Exploration: scene complexity")
    b.table([t("Mesure", "Measure"), t("Base", "KB"), "Validation", t("Test initial", "Initial test")], [
        [t("Annotations / image : moyenne", "Annotations/image: mean"), "7.2507", "7.3400", "7.0500"],
        [t("Médiane", "Median"), "4", "4.5", "4"],
        [t("Écart-type de population", "Population standard deviation"), "7.3305", "8.3848", "6.4628"],
        [t("Minimum - maximum", "Minimum - maximum"), "0 - 59", "0 - 58", "1 - 26"]])
    b.figure("knowledge_base_objects_per_image.png", 190,
             "Distribution du nombre d'annotations par image dans la base.",
             "Distribution of annotation counts per knowledge-base image.")
    b.figure("knowledge_base_complexity_caption_scatter.png", 180,
             "Chaque point relie le nombre d'annotations et la longueur moyenne des cinq captions d'une image.",
             "Each point links annotation count to the mean length of an image's five captions.")
    b.p("Les scènes simples représentent 4 247 images (42,47 %), les moyennes 2 449 (24,49 %) et les complexes 3 304 (33,04 %). Ces noms sont des règles de comptage choisies pour l'échantillonnage, pas des labels officiels de COCO. Une image sans annotation n'est pas nécessairement vide. La moyenne supérieure à la médiane reflète la présence de scènes très chargées ; le nuage de points ne prouve pas de relation causale avec la longueur des captions.",
        "Simple scenes account for 4,247 images (42.47%), medium scenes for 2,449 (24.49%) and complex scenes for 3,304 (33.04%). These are count-based sampling rules, not official COCO labels. An image with no annotations is not necessarily empty. A mean above the median reflects crowded scenes; the scatter plot does not establish a causal relationship with caption length.")

    b.page("9. Exploration : texte et vocabulaire", "9. Exploration: text and vocabulary")
    b.table([t("Mesure", "Measure"), t("Base", "KB"), "Validation", t("Test initial", "Initial test")], [
        [t("Captions", "Captions"), "50 000", "500", "500"],
        [t("Mots / caption : moyenne", "Words/caption: mean"), "10.4525", "10.2900", "10.2720"],
        [t("Médiane", "Median"), "10", "10", "10"],
        [t("Mots / caption : min-max", "Words/caption: min-max"), "6-49", "7-32", "7-21"],
        [t("Vocabulaire distinct", "Distinct vocabulary"), "9 972", "945", "904"],
        [t("Mots comptés au total", "Total counted words"), "522 625", "5 145", "5 136"]])
    b.figure("knowledge_base_pos.png", 210,
             "Distribution des catégories grammaticales Penn Treebank sur les captions brutes de la base.",
             "Penn Treebank POS distribution for the knowledge base's raw captions.")
    b.p("NN correspond aux noms communs singuliers (149 690 tokens), DT aux déterminants (107 020), IN aux prépositions ou conjonctions de subordination (83 586), JJ aux adjectifs (41 620) et VBG aux formes en -ing (29 990). Il s'agit de prédictions du tagger anglais NLTK, pas d'étiquettes grammaticales vérifiées à la main.",
        "NN denotes singular common nouns (149,690 tokens), DT determiners (107,020), IN prepositions or subordinating conjunctions (83,586), JJ adjectives (41,620) and VBG -ing forms (29,990). These are predictions from NLTK's English tagger, not manually verified grammatical labels.")
    b.p("Le vocabulaire de la base est plus grand en partie parce qu'elle contient cent fois plus de captions. Comparer directement 9 972 et 945 ne démontre donc pas une meilleure diversité. Les histogrammes de longueur, fréquences des mots et tables POS sont également sauvegardés pour chaque jeu dans reports/figures/ et reports/tables/.",
        "The knowledge base has a larger vocabulary partly because it contains one hundred times as many captions. Comparing 9,972 with 945 does not establish greater diversity by itself. Caption-length histograms, word frequencies and POS tables are saved for each split in reports/figures/ and reports/tables/.")

    b.page("10. Cooccurrences, relations et diversité", "10. Co-occurrence, relations and diversity")
    b.figure("knowledge_base_cooccurrence.png", 310,
             "Cooccurrence des 20 catégories les plus fréquentes : une case compte les images où les deux catégories apparaissent. La diagonale indique le nombre d'images contenant la catégorie.",
             "Co-occurrence of the 20 most frequent categories: each cell counts images containing both categories. The diagonal counts images containing each category.")
    b.p("La cooccurrence travaille sur la présence d'une catégorie par image : dix personnes dans une photo ne valent qu'une présence de person. Elle ne mesure pas une interaction, par exemple tenir ou conduire. Les relations left_of et above comparent les centres des boîtes ; overlap utilise une intersection sur union (IoU) d'au moins 0,1. Il y a 495 256 paires orientées left_of et 45 020 paires overlap dans la base. Les directions inverses se répondent ; il ne faut pas additionner ces comptes comme des relations indépendantes.",
        "Co-occurrence uses image-level category presence: ten people in a photograph count as one presence of person. It does not measure interactions such as holding or driving. left_of and above compare box centres; overlap requires intersection over union (IoU) of at least 0.1. The base has 495,256 ordered left_of pairs and 45,020 overlap pairs. Opposite directions mirror one another; these counts are not independent relations to be summed.")
    b.p("Pour la diversité, un petit lexique cherche les mots de couleur, de forme et de contexte dans les cinq captions d'une image. white apparaît dans 1 997 images, red dans 870, night dans 136 et round dans 61. Ce sont des mentions textuelles : white ne dit pas quel objet est blanc et l'absence de night ne signifie pas day. Le mot on n'est pas une relation visuelle vérifiée.",
        "For diversity, a small lexicon searches colour, shape and context words in each image's five captions. white appears for 1,997 images, red for 870, night for 136 and round for 61. These are textual mentions: white does not identify which object is white, and missing night does not mean day. The word on is not a verified visual relationship.")
    b.p("Les 33 figures initiales couvrent 11 visualisations par jeu. Si un détecteur est ajouté, de nouvelles statistiques devront être produites sur ses propres sorties et distinguées de ces annotations et heuristiques.",
        "The initial 33 figures cover 11 visualisations per split. If a detector is added, new statistics must describe its actual outputs and remain distinct from these annotations and heuristics.")

    b.page("11. Embeddings CLIP et index FAISS", "11. CLIP embeddings and FAISS indexes")
    b.p("Un embedding est une liste de nombres représentant le contenu d'une image ou d'un texte. CLIP fournit ici 512 nombres pour chaque image et chaque caption. Des contenus proches peuvent recevoir des vecteurs proches. CLIP est figé en mode inférence ; il ne rédige pas de captions dans cette phase.",
        "An embedding is a list of numbers representing image or text content. Here CLIP produces 512 values per image and per caption. Similar content may have nearby vectors. CLIP is frozen in inference mode and does not write captions in this phase.")
    b.code("Model: openai/clip-vit-base-patch32\nRevision: " + em["resolved_revision"] + "\nDevice: CPU | batch: 32 | dtype: float32 | L2 normalisation")
    b.table([t("Artefact", "Artifact"), t("Forme / nombre", "Shape / count"), t("Fichier", "File")], [
        [t("Vecteurs image", "Image vectors"), "10 000 x 512", "artifacts/embeddings/image.npy"],
        [t("Vecteurs texte", "Text vectors"), "50 000 x 512", "artifacts/embeddings/text.npy"],
        [t("Index image", "Image index"), "10 000", "artifacts/indexes/image.index"],
        [t("Index texte", "Text index"), "50 000", "artifacts/indexes/text.index"]])
    b.p("Les vecteurs sont divisés par leur norme L2 pour avoir une longueur de 1. FAISS utilise IndexFlatIP : il compare la requête à tous les vecteurs de l'index par produit scalaire. Sur ces vecteurs normalisés, ce produit est la similarité cosinus. L'index est exact pour les vecteurs fournis ; il ne garantit pas que le voisin représente le même objet.",
        "Vectors are divided by their L2 norm to obtain length 1. FAISS uses IndexFlatIP: it compares the query with every stored vector using inner product. With normalised vectors this equals cosine similarity. Search is exact for the supplied vectors; this does not guarantee that a neighbour depicts the same object.")
    b.table([t("Index", "Index"), t("Construction enregistrée", "Recorded build time"), t("Taille disque", "Disk bytes")], [
        [k, f"{v['build_seconds']:.4f} s", str(v['bytes_on_disk'])] for k, v in indexes.items()])
    b.p("Les fichiers image_rows.json et text_rows.json assurent la correspondance entre position du vecteur, image_id et caption_id. Les métadonnées contiennent ensuite les phrases et objets. Les contrôles reconstruisent les vecteurs FAISS et les comparent aux matrices sauvegardées. Seuls les 10 000 IDs de la base alimentent les deux index.",
        "image_rows.json and text_rows.json map vector positions to image_id and caption_id. Metadata then provide captions and objects. Checks reconstruct FAISS vectors and compare them with the saved matrices. Only the 10,000 knowledge-base IDs feed either index.")
    b.p("Résultat : matrices complètes, sans NaN ni infini, normalisées et sans IDs de lignes dupliqués ; aucune caption tronquée par CLIP. Le calcul a utilisé des checkpoints par lot. Les 582 secondes du dernier lancement reprennent des lots déjà calculés : ce chiffre n'est pas le temps total depuis zéro.",
        "Result: complete, finite, normalised matrices with no duplicate row IDs, and no caption truncated by CLIP. Batch checkpoints supported resuming. The last invocation's 582 seconds include reuse of prior batches; this is not the total computation time from scratch.")

    b.page("12. Vérifier la recherche d'images proches", "12. Checking similar-image retrieval")
    b.p("Pour chacune des 100 images de validation, la photographie seule est prétraitée puis encodée par CLIP. Son vecteur interroge l'index image de la base. Le programme récupère les cinq voisins, leurs IDs, leurs scores et leurs captions de référence. Les phrases de validation n'interviennent pas dans la requête.",
        "Each of the 100 validation photographs alone is pre-processed and encoded with CLIP. Its vector queries the knowledge-base image index. The program retrieves five neighbours, their IDs, scores and reference captions. Validation captions do not enter the query.")
    b.figure("../retrieval/manual_490_k5/nearest_images.png", 225,
             "Recherche réelle : image de validation 490 et ses cinq voisins. Les textes affichés sont des références COCO récupérées, pas des captions générées.",
             "Actual retrieval: validation image 490 and its five neighbours. Displayed text consists of retrieved COCO references, not generated captions.")
    q = retrieval["queries"][0]
    b.table([t("Rang", "Rank"), "COCO image ID", t("Similarité cosinus", "Cosine similarity")], [
        [str(i + 1), str(h["image_id"]), f"{h['cosine_similarity']:.6f}"] for i, h in enumerate(q["results"])])
    b.p(f"Les 100 requêtes ont produit 500 voisins sauvegardés en JSON et CSV. Le temps enregistré d'encodage des lots est {retrieval['encoding_seconds']:.3f} s ; la recherche moyenne par requête est {retrieval['mean_search_ms']:.3f} ms sur cette machine. Ce sont des observations fonctionnelles, pas un benchmark matériel contrôlé. Les 500 résultats ont été rejoués contre FAISS lors de la revue.",
        f"The 100 queries produced 500 neighbours saved as JSON and CSV. Recorded batch encoding time is {retrieval['encoding_seconds']:.3f} s; mean search time is {retrieval['mean_search_ms']:.3f} ms per query on this machine. These are functional observations, not a controlled hardware benchmark. All 500 results were replayed against FAISS during review.")
    b.p("Un score de 0,818 ne signifie pas 81,8 % de probabilité d'une caption correcte. Sans labels de pertinence ni captions générées, nous ne calculons ici ni Recall@k, ni CIDEr, ni BLEU. La proximité visuelle peut être imparfaite ; le test vérifie le fonctionnement et la traçabilité de la recherche.",
        "A score of 0.818 is not an 81.8% probability of a correct caption. Without relevance labels or generated captions, Recall@k, CIDEr and BLEU are not calculated here. Visual similarity may be imperfect; this check establishes operation and traceability.")

    b.page("13. Reproduire : installer et préparer", "13. Reproducing: installation and data preparation")
    b.p("Dans VS Code, ouvrir le dossier du projet, puis Terminal > New Terminal. Les commandes ci-dessous utilisent PowerShell sous Windows. Elles reproduisent la version actuelle à 100 images de test, pas le futur jeu de 1 000. Lancer les commandes une par une et s'arrêter si une commande échoue.",
        "In VS Code, open the project folder and choose Terminal > New Terminal. Commands below use Windows PowerShell. They reproduce the current 100-image test release, not the future 1,000-image set. Run commands one at a time and stop if any command fails.")
    b.h("A. Sur un nouveau PC, après téléchargement du code", "A. On a new PC, after downloading the source")
    b.p("Installer Python 3.12 x64 puis créer l'environnement. L'exécution historique a utilisé Python 3.12.10 et des bibliothèques présentes sur ce PC ; ce n'était pas une installation isolée. Les versions exactes sont enregistrées dans les manifestes. Une installation propre sur une autre machine doit être vérifiée séparément.",
        "Install Python 3.12 x64, then create the environment. The historical run used Python 3.12.10 and libraries already present on this PC, rather than a fully isolated installation. Exact versions are in the manifests. A clean installation on another machine needs separate validation.")
    b.code("py -3.12 -m venv imp\n.\\imp\\Scripts\\python.exe -m pip install -r requirements.txt\n$impPython = \".\\imp\\Scripts\\python.exe\"")
    b.h("B. Récupérer les entrées nécessaires", "B. Download the required inputs")
    b.code("& $impPython -m src.cli download-annotations\n& $impPython -m src.cli download-images\n& $impPython -m src.cli download-nlp\n& $impPython -m src.cli download-model\n& $impPython -m src.cli audit")
    b.p("Réutiliser les listes artifacts/splits/ fournies avec le projet. Ne pas relancer splits --force pour une simple reproduction. download-images utilise ces IDs et ignore les fichiers déjà présents ; audit doit signaler toute image illisible. download-model charge les poids CLIP à la révision fixée, tandis que download-nlp récupère le modèle grammatical anglais.",
        "Reuse the supplied artifacts/splits/ lists. Do not run splits --force for routine reproduction. download-images uses those IDs and skips existing files; audit should report unreadable images. download-model loads CLIP weights at the pinned revision, while download-nlp obtains the English grammatical tagger.")
    b.h("C. Préparer les métadonnées et les statistiques", "C. Prepare metadata and statistics")
    b.code("& $impPython -m src.cli metadata\n& $impPython -m src.cli eda")
    b.p("Sorties attendues : trois JSONL et un manifeste dans artifacts/metadata/, reports/eda.json, ainsi que les figures et tables. Les originaux data/raw/ restent inchangés. Dans le projet déjà préparé, ces étapes n'ont pas besoin d'être répétées pour une démonstration de recherche.",
        "Expected outputs: three JSONL files and a manifest in artifacts/metadata/, reports/eda.json, plus figures and tables. data/raw/ originals remain unchanged. In the already prepared project, these steps need not be repeated for a retrieval demonstration.")

    b.page("14. Reproduire : indexer et tester soi-même", "14. Reproducing: indexing and manual testing")
    b.h("D. Calculer puis contrôler les artefacts", "D. Compute and check the artifacts")
    b.code("$impPython = \".\\imp\\Scripts\\python.exe\"\n& $impPython -m src.cli --offline embeddings\n& $impPython -m src.cli index\n& $impPython -m src.cli --offline validate-retrieval --top-k 5\n& $impPython -m src.cli --offline qc\n& $impPython -m src.cli report")
    b.p("Le calcul CLIP est l'étape coûteuse sur CPU. Si les matrices et leurs empreintes sont valides, embeddings les réutilise. Les checkpoints permettent de reprendre un calcul incomplet. qc vérifie les données, les références, les correspondances, les valeurs des vecteurs et les résultats sauvegardés. report produit le résumé technique d'exécution.",
        "CLIP computation is the expensive CPU step. embeddings reuses valid matrices after checking their fingerprints. Checkpoints allow incomplete runs to resume. qc checks data, references, mappings, vector values and saved retrieval results. report produces the technical execution summary.")
    b.h("E. Choisir une image et le nombre de voisins", "E. Choose an image and neighbour count")
    b.code("& $impPython -m src.cli --offline retrieve `\n    --image-id 490 --top-k 5 --show\n& $impPython -m src.cli --offline retrieve `\n    --image-id 133 --top-k 10 --show\n& $impPython -m src.cli --offline retrieve `\n    --image-path \"C:\\photos\\ma_photo.jpg\" --top-k 5 --show")
    b.p("Remplacer le chemin de la dernière commande par un fichier qui existe. Les IDs COCO ne sont pas des numéros de ligne : ID 133 correspond au fichier 000000000133.jpg. Un ID choisi au hasard peut être absent des annotations ou du sous-ensemble téléchargé. Pour voir les photos, ouvrir data/raw/train2017/ en affichage grandes icônes. La photo de validation 490 est un exemple disponible.",
        "Replace the final command's path with an existing file. COCO IDs are not row numbers: ID 133 corresponds to 000000000133.jpg. An arbitrary ID may be absent from the annotations or downloaded subset. Browse data/raw/train2017/ with large icons to see photographs. Validation image 490 is an available example.")
    b.p("Chaque recherche écrit results.json, results.csv, gallery.html et nearest_images.png dans reports/retrieval/. Une image de la base utilisée comme requête est exclue de ses propres résultats. Le ZIP Windows complet permet aussi de lancer TESTER.cmd sans reconstruire les artefacts ; le ZIP source GitHub ne contient pas ces gros fichiers.",
        "Each search writes results.json, results.csv, gallery.html and nearest_images.png under reports/retrieval/. A knowledge-base image used as a query is excluded from its own results. The complete Windows ZIP also provides TESTER.cmd without rebuilding artifacts; GitHub's source ZIP does not include those large files.")
    b.code("& $impPython scripts/review_phase1.py\n& $impPython scripts/build_phase1_report.py")
    b.p("La dernière commande reconstruit ce rapport avec reportlab et pypdf, installables dans tmp/report_deps comme indiqué en tête du script. Ces deux bibliothèques servent uniquement au rapport.",
        "The last command rebuilds this report with reportlab and pypdf, installable into tmp/report_deps as documented at the top of the script. These libraries are only needed for reporting.")

    b.page("15. Bilan, limites et éléments à compléter", "15. Status, limitations and remaining work")
    b.p("Le travail exécuté fournit une base vérifiée de 10 000 images et 50 000 captions, 100 images de validation distinctes, les métadonnées, les statistiques, les embeddings et deux index FAISS. Les contrôles qualité passent pour la version configurée. Ce succès technique ne signifie pas que la nouvelle cible de test est déjà satisfaite.",
        "Executed work provides a verified 10,000-image, 50,000-caption knowledge base, 100 distinct validation images, metadata, statistics, embeddings and two FAISS indexes. Quality checks pass for the configured release. This technical success does not mean that the updated test target is already met.")
    b.h("Suite nécessaire pour la phase 1", "Remaining phase-1 work")
    b.p("1. Conserver les 100 anciens IDs et leurs résultats. Préparer et figer séparément 1 000 IDs de val2017, communs aux groupes si la liste est fournie ; documenter leur éventuelle intersection avec les 100 premiers. Télécharger, auditer et préparer leurs métadonnées et statistiques. Ne pas recalculer la base uniquement parce que le test change.",
        "1. Preserve the original 100 IDs and their results. Separately prepare and freeze 1,000 val2017 IDs, common across groups if supplied; document any overlap with the initial 100. Download, audit and prepare their metadata and statistics. A test-set change alone should not require recomputing the knowledge base.")
    b.p("2. Documenter explicitement la convention de confiance des références humaines. Si une extraction visuelle est ajoutée, fixer le détecteur et son seuil, exécuter l'extraction puis produire des statistiques séparées. Cette option recommandée ne doit pas être annoncée comme exécutée tant qu'elle ne l'est pas.",
        "2. Explicitly document the human-reference confidence convention. If visual extraction is added, fix the detector and threshold, execute extraction, then produce separate statistics. This recommended option must not be reported as executed before it actually is.")
    b.p("3. Faire valider la préparation par l'enseignante. Aucune configuration optimale n'est démontrée par le seul test fonctionnel. Les labels de contexte sont heuristiques ; les annotations de foule ne donnent pas un recensement exact ; le recadrage peut perdre du contenu. La comparaison de performances entre tests 100 et 1 000 attendra une évaluation définie dans la phase suivante.",
        "3. Obtain teacher approval of the preparation. Functional checks alone do not establish an optimal configuration. Context labels are heuristic; crowd annotations are not an exact census; cropping may lose content. A performance comparison between 100- and 1,000-image tests requires an evaluation protocol in the next phase.")
    b.h("Sources et preuves à consulter", "Sources and evidence")
    b.p("[1] Cahier des charges officiel : docs/project/Image processing mini-project english version.pdf, sections 4.1-4.4 et 8. [2] Précisions transmises par l'étudiant : docs/project/teacher_clarifications_20261107.md. [3] Méthode exécutée : src/data/, src/preprocessing/, src/exploration/, src/embeddings/ et src/indexing/.",
        "[1] Official specification: docs/project/Image processing mini-project english version.pdf, sections 4.1-4.4 and 8. [2] Clarifications relayed by the student: docs/project/teacher_clarifications_20261107.md. [3] Executed implementation: src/data/, src/preprocessing/, src/exploration/, src/embeddings/ and src/indexing/.")
    b.p("[4] Effectifs et anomalies : reports/audit.json. [5] Statistiques et tableaux : reports/eda.json et reports/tables/. [6] Modèle, dimensions et index : artifacts/embeddings/manifest.json et artifacts/indexes/manifest.json. [7] Recherche : reports/retrieval/validation/results.json. [8] Vérification actuelle : reports/quality_control.json et reports/phase1_review_20261107.json.",
        "[4] Counts and anomalies: reports/audit.json. [5] Statistics and tables: reports/eda.json and reports/tables/. [6] Model, dimensions and indexes: artifacts/embeddings/manifest.json and artifacts/indexes/manifest.json. [7] Retrieval: reports/retrieval/validation/results.json. [8] Current verification: reports/quality_control.json and reports/phase1_review_20261107.json.")
    b.p("Sources publiques des composants : COCO (cocodataset.org), modèle openai/clip-vit-base-patch32 sur Hugging Face et bibliothèque FAISS (github.com/facebookresearch/faiss). Les chiffres de ce rapport proviennent exclusivement des artefacts locaux décrits ci-dessus.",
        "Public component sources: COCO (cocodataset.org), openai/clip-vit-base-patch32 on Hugging Face, and FAISS (github.com/facebookresearch/faiss). All numerical results in this report come exclusively from the local artifacts listed above.")
    return extended_release_content(b, eda, review)


def extended_release_content(b, eda, review):
    """Bring every report section into line with the verified test1000 release."""
    t = b.tr
    audit = read_json(ROOT / "reports/audit.json")
    shared = len(audit["shared_caption_details"])
    b.pages[0]["blocks"] = [
        ("center", t("République algérienne démocratique et populaire\nMinistère de l'Enseignement supérieur et de la Recherche scientifique",
                     "People's Democratic Republic of Algeria\nMinistry of Higher Education and Scientific Research")),
        ("center", t("Université d'Alger 1 Benyoucef Benkhedda\nFaculté des sciences - Département d'informatique",
                     "University of Algiers 1 Benyoucef Benkhedda\nFaculty of Sciences - Department of Computer Science")),
        ("logo", None),
        ("center", t("Master 1 - Ingénierie des systèmes informatiques intelligents\nModule : Introduction au Traitement d'Images",
                     "Master 1 - Intelligent Computer Systems Engineering\nModule: Introduction to Image Processing")),
        ("cover_title", "A comparative study of RAG techniques\nfor generating image captions"),
        ("center", t("Rapport de phase 1", "Phase-1 report")),
        ("cover_title", b.pages[0]["title"]),
        ("center", t("Réalisé par", "Prepared by")),
        ("center", "Hamou Yasmine\nOuziala Ali\nLounis Aya"),
        ("center", t("Année universitaire 2026-2027\nÉchéance de la phase : 4 novembre 2026\nVersion mise à jour : 7 novembre 2026",
                     "Academic year 2026-2027\nPhase deadline: 4 November 2026\nUpdated version: 7 November 2026")),
    ]
    # Dataset: legacy 100 is a nested analysis view, not an extra split/image count.
    blocks = b.pages[2]["blocks"]
    for j, (kind, value) in enumerate(blocks):
        if kind == "table":
            headers, rows = value
            rows[-1][2] = "1 000"
        if kind == "p" and (value.startswith("10 000 +") or value.startswith("10,000 +")):
            blocks[j] = (kind, t("10 000 + 100 = 10 100 images distinctes de train2017. Les 1 000 images de test viennent de val2017 : le total est donc de 11 100 images distinctes. Les 100 images du premier test sont incluses dans les 1 000 ; elles ne s'ajoutent pas une seconde fois.",
                                "10,000 + 100 = 10,100 distinct train2017 images. The 1,000 test images come from val2017, giving 11,100 unique selected images in total. The original 100 test images are included within the 1,000, not counted again."))
        if kind == "p" and (value.startswith("Les listes sont") or value.startswith("Lists are frozen")):
            blocks[j] = (kind, t("La commande extend-test a conservé exactement les IDs de la base et de la validation. Elle a gardé les 100 anciens IDs et ajouté un tirage stratifié de 900 autres IDs éligibles de val2017, avec la graine 42. Les listes sont gelées et vérifiées par SHA256. L'ancienne version est archivée dans reports/releases/. La liste commune de la classe n'ayant pas été fournie, ce test local ne peut pas être présenté comme le test commun officiel.",
                                "extend-test preserved the exact KB and validation IDs, retained the original 100 test IDs and added a seed-42 stratified sample of 900 other eligible val2017 IDs. Lists are frozen and SHA256-checked. The previous release is archived under reports/releases/. As the class-wide ID list was not supplied, this local test must not be presented as the official common test."))
    for j, (kind, value) in enumerate(b.pages[3]["blocks"]):
        if kind == "table":
            value[1][0][1] = "11 100 / 11 100 / 11 100"
            value[1][-1][1] = str(shared)
            value[1][-2][1] = str(sum(x["kind"] == "degenerate_bbox" for x in audit["issues"]))
        if kind == "p" and (value.startswith("Les 19") or value.startswith("The 19")):
            b.pages[3]["blocks"][j] = (kind, t(f"L'audit étendu relève {shared} textes normalisés partagés entre jeux. Ces phrases appartiennent à des images et des annotations distinctes : ce signal est documenté sans suppression automatique. Les doublons exacts de fichiers et de pixels sont vérifiés ; les quasi-doublons visuels restent une limite. Les preuves sont dans reports/audit.json et reports/tables/.",
                f"The extended audit identifies {shared} normalised caption strings shared across splits. They belong to distinct images and annotations and are documented without automatic deletion. Exact file and pixel duplicates are checked; perceptual near-duplicates remain a limitation. Evidence is in reports/audit.json and reports/tables/."))
    b.pages[4]["blocks"][-1] = ("p", t("Chaque caption humaine porte maintenant source=coco_human_reference et confidence=1. Le seuil de conservation des références est également fixé à 1 dans la configuration et dans la politique des métadonnées : les cinq références humaines sont donc gardées. Cette valeur est une convention de provenance, pas une probabilité calibrée. Elle n'est attribuée ni aux attributs repérés par lexique ni aux relations déduites. Aucun seuil de détecteur visuel ne s'applique, puisqu'aucun détecteur n'a été exécuté.",
        "Each human caption now carries source=coco_human_reference and confidence=1. The reference retention threshold is also 1 in the configuration and metadata policy: all five human references are retained. This is a provenance convention, not a calibrated probability. It is not assigned to lexicon-derived attributes or inferred relations. No visual detector threshold applies because no detector was run."))
    # Side-by-side statistics for the first 100 and the extended 1000.
    stat_splits = ["knowledge_base", "validation", "test_100", "test"]
    labels = [t("Mesure", "Measure"), t("Base", "KB"), "Validation", "Test 100", "Test 1000"]
    complexity_rows = []
    for label, key, fmt in [(t("Annotations/image : moyenne", "Annotations/image: mean"), "mean", ".4f"),
                            (t("Médiane", "Median"), "median", ".1f"),
                            (t("Écart-type", "Standard deviation"), "std_population", ".4f")]:
        complexity_rows.append([label] + [format(eda[s]["objects_per_image"][key], fmt) for s in stat_splits])
    complexity_rows.append([t("Minimum - maximum", "Minimum - maximum")] + [f"{eda[s]['objects_per_image']['min']} - {eda[s]['objects_per_image']['max']}" for s in stat_splits])
    for j, (kind, value) in enumerate(b.pages[8]["blocks"]):
        if kind == "table": b.pages[8]["blocks"][j] = (kind, (labels, complexity_rows))
        if kind == "figure" and "scatter" in value[0]:
            b.pages[8]["blocks"][j] = (kind, ("phase1_test100_test1000.png", 170, t("Comparaison descriptive des tests 100 et 1 000 : fréquences normalisées des nombres d'objets et longueurs de captions.", "Descriptive comparison of test100 and test1000: normalised object-count and caption-length frequencies.")))
    b.pages[8]["blocks"][-1] = ("p", t("La base comprend 4 247 scènes simples, 2 449 moyennes et 3 304 complexes selon les seuils de comptage choisis. Ces catégories ne sont pas des labels officiels de COCO. La moyenne supérieure à la médiane signale des scènes très chargées. Le test de 100 images étant inclus dans celui de 1 000, les deux échantillons ne sont pas indépendants. La comparaison décrit les données ; elle ne mesure pas une performance du RAG.",
        "The KB contains 4,247 simple, 2,449 medium and 3,304 complex scenes under the selected count thresholds, not official COCO scene labels. A mean above the median reflects crowded scenes. Test100 is nested within test1000, so they are not independent samples. This comparison describes the data rather than RAG performance."))
    text_rows = [[t("Captions", "Captions")] + [str(eda[s]["captions"]) for s in stat_splits],
        [t("Mots/caption : moyenne", "Words/caption: mean")] + [f"{eda[s]['caption_length_words']['mean']:.4f}" for s in stat_splits],
        [t("Médiane", "Median")] + [str(eda[s]["caption_length_words"]["median"]) for s in stat_splits],
        [t("Minimum - maximum", "Minimum - maximum")] + [f"{eda[s]['caption_length_words']['min']}-{eda[s]['caption_length_words']['max']}" for s in stat_splits],
        [t("Vocabulaire distinct", "Distinct vocabulary")] + [str(eda[s]["vocabulary_size"]) for s in stat_splits],
        [t("Mots au total", "Total words")] + [str(eda[s]["word_tokens"]) for s in stat_splits]]
    for j, (kind, value) in enumerate(b.pages[9]["blocks"]):
        if kind == "table": b.pages[9]["blocks"][j] = (kind, (labels, text_rows))
    b.pages[10]["blocks"][-1] = ("p", t("44 figures décrivent les quatre vues : base, validation, test de 1 000 images et sous-ensemble historique de 100 images. La figure comparative supplémentaire facilite la lecture des deux tests. Les statistiques des objets proviennent des annotations COCO ; celles d'un éventuel détecteur devront être calculées séparément sur ses sorties réelles.",
        "44 figures describe four views: KB, validation, the 1,000-image test and the original 100-image subset. An additional comparison figure helps inspect the two tests. Object statistics come from COCO annotations; any future detector's statistics must separately describe its actual outputs."))
    b.pages[11]["blocks"][-1] = ("p", t("Les matrices sont complètes, finies et normalisées, avec des IDs de lignes uniques ; aucune caption n'a été tronquée par CLIP. Lors de l'extension du test et de l'ajout des champs de confiance, les véritables entrées de l'encodeur ont été comparées à l'ancienne version : les images et textes de la base sont identiques. Les fichiers .npy ont donc été conservés, empreintes inchangées, et les index ont été reconstruits. Cette migration vérifiée est enregistrée dans reports/phase1_upgrade.json ; aucun nouvel entraînement n'a été réalisé.",
        "Matrices are complete, finite and normalised with unique row IDs; no caption was truncated by CLIP. When extending the test and adding confidence fields, actual encoder inputs were compared with the previous release: KB images and texts are identical. The .npy files were retained with unchanged hashes and the indexes rebuilt. This verified migration is recorded in reports/phase1_upgrade.json; no new training was performed."))
    b.pages[13]["blocks"][0] = ("p", t("Dans VS Code, ouvrir le dossier du projet puis Terminal > New Terminal. Ces commandes PowerShell reproduisent la version actuelle : 10 000 images de base, 100 de validation distinctes et 1 000 de test, dont les 100 premières restent identifiables. Les IDs sont déjà fournis : ne pas refaire le tirage. Exécuter les commandes une par une et s'arrêter si une commande échoue.",
        "In VS Code, open the project folder and choose Terminal > New Terminal. These PowerShell commands reproduce the current release: 10,000 KB, 100 separate validation and 1,000 test images, with the original 100 identifiable. Frozen IDs are supplied; do not resample. Run commands one at a time and stop on failure."))
    for j, (kind, value) in enumerate(b.pages[14]["blocks"]):
        if kind == "p" and (value.startswith("Chaque recherche écrit") or value.startswith("Each search writes")):
            b.pages[14]["blocks"][j] = (kind, t("Chaque recherche écrit results.json, results.csv, gallery.html et nearest_images.png dans reports/retrieval/. Une image de la base utilisée comme requête est exclue de ses propres résultats. Le dossier Projet_ITI_Drive_v2 contient le code et les données préparées dans trois ZIP : restaurer_donnees.py les remet aux bons chemins. Installer ensuite Python et les bibliothèques une fois, selon son README ; les embeddings ne sont pas recalculés.",
                "Each search writes results.json, results.csv, gallery.html and nearest_images.png under reports/retrieval/. A KB image used as a query is excluded from its own results. Projet_ITI_Drive_v2 includes the code and prepared data in three ZIPs; restaurer_donnees.py restores the expected paths. Install Python and the libraries once following its README; embeddings are not recomputed."))
    last = b.pages[15]["blocks"]
    last[0] = ("p", t("La préparation fournit maintenant 10 000 images de base, 100 de validation distinctes et 1 000 de test, soit 11 100 images uniques. Le premier test de 100 est conservé dans le test étendu et analysé séparément. Les références humaines portent la convention de confiance à 1. Les métadonnées, statistiques, vecteurs et index ont été contrôlés sur les données réelles.",
        "Preparation now provides 10,000 KB, 100 separate validation and 1,000 test images: 11,100 unique images. The original test100 is retained within the extended test and analysed separately. Human references carry the confidence-1 convention. Metadata, statistics, vectors and indexes have been checked on real data."))
    last[1] = ("h", t("Décisions, limites et validation", "Decisions, limitations and approval"))
    last[2] = ("p", t("1. La cible de 1 000 images est atteinte sans modifier les IDs de la base ou de la validation. Les anciennes données et preuves sont archivées. Le test est un échantillon local stratifié : sa conformité à une liste commune de la classe ne peut pas être certifiée tant que cette liste n'est pas fournie. L'enseignante doit valider la préparation.",
        "1. The 1,000-image target is met without changing KB or validation IDs. Previous data and evidence are archived. The test is locally stratified; conformity to a common class-wide list cannot be certified until that list is supplied. Teacher approval is still required."))
    last[3] = ("p", t("2. La méthode retenue utilise les annotations COCO et conserve la source de chaque champ. La détection dans les pixels reste une option recommandée, non exécutée ici. Les couleurs et contextes issus des captions restent heuristiques. La confiance 1 des références humaines ne leur est pas transférée.",
        "2. The selected method uses COCO annotations and records each field's source. Pixel-based detection remains a recommended option and was not run here. Caption-derived colours and contexts remain heuristic and do not inherit the human-reference confidence of 1."))
    from scripts.report_pedagogy import enhance_book
    return enhance_book(b)


def draw_book(b):
    fontroot = Path("C:/Windows/Fonts")
    for name, filename in [("Body", "arial.ttf"), ("BodyBold", "arialbd.ttf"), ("Code", "consola.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fontroot / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    blue = colors.HexColor("#233e59")
    styles = {
        "title": ParagraphStyle("title", fontName="BodyBold", fontSize=18, leading=23, spaceAfter=17, textColor=blue),
        "cover": ParagraphStyle("cover", fontName="BodyBold", fontSize=26, leading=32, spaceAfter=24, textColor=blue),
        "p": ParagraphStyle("p", fontName="Body", fontSize=10, leading=14.3, spaceAfter=10),
        "h": ParagraphStyle("h", fontName="BodyBold", fontSize=11.5, leading=16, spaceBefore=7, spaceAfter=8, textColor=blue),
        "small": ParagraphStyle("small", fontName="Body", fontSize=8.2, leading=11, spaceAfter=9, textColor=colors.HexColor("#495465")),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.4, leading=11),
        "code": ParagraphStyle("code", fontName="Code", fontSize=8.2, leading=11.8, spaceAfter=12,
                               borderPadding=9, backColor=colors.HexColor("#f0f3f6")),
        "center": ParagraphStyle("center", fontName="Body", fontSize=10.5, leading=15, alignment=TA_CENTER, spaceAfter=13),
        "cover_title": ParagraphStyle("cover_title", fontName="BodyBold", fontSize=17, leading=23, alignment=TA_CENTER, textColor=blue, spaceBefore=8, spaceAfter=16),
    }
    def p(text, style="p"):
        text = escape(str(text))
        # Permit long paths to wrap instead of escaping the right margin.
        text = text.replace("/", "/<wbr/>").replace("_", "_<wbr/>").replace("\n", "<br/>")
        return Paragraph(text, styles[style])
    story = []
    filename = OUT / ("Rapport_phase1_FR.pdf" if b.lang == "fr" else "Phase1_report_EN.pdf")
    md = ["# " + b.pages[0]["title"], ""]
    content_width = A4[0] - 100
    for page_number, page in enumerate(b.pages):
        if page_number: story.append(PageBreak())
        if page_number == 0: story.append(Spacer(1, 8))
        else: story.append(p(page["title"], "title"))
        if page_number: md += ["## " + page["title"], ""]
        for kind, value in page["blocks"]:
            if kind in ("p", "h", "center", "cover_title"):
                story.append(p(value, kind))
                md += [("### " if kind == "h" else "") + value, ""]
            elif kind == "logo":
                logo = ROOT / "docs/assets/universite_alger1_logo.png"
                story += [Image(str(logo), width=70, height=70), Spacer(1, 14)]
                md += ["![Université d'Alger 1](../../docs/assets/universite_alger1_logo.png)", ""]
            elif kind == "code":
                assert max(len(line) for line in value.splitlines()) <= 94
                story.append(Preformatted(value, styles["code"]))
                md += ["```powershell", value, "```", ""]
            elif kind == "table":
                header, rows = value
                sizes = [content_width / len(header)] * len(header)
                data = [[p(c, "cell") for c in row] for row in [header] + rows]
                table = Table(data, colWidths=sizes, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7edf3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), .7, blue),
                    ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#ccd4dc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]))
                story += [table, Spacer(1, 12)]
                md += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                md += ["| " + " | ".join(map(str, row)) + " |" for row in rows] + [""]
            elif kind == "figure":
                name, limit, caption = value
                target = FIG / name
                with PILImage.open(target) as img: width, height = img.size
                scale = min(content_width / width, limit / height)
                story += [Image(str(target), width=width * scale, height=height * scale), Spacer(1, 5), p(caption, "small")]
                md += [f"![{caption}](../../reports/figures/{name})", ""]
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#ccd4dc")); canvas.line(50, 43, A4[0]-50, 43)
        canvas.setFont("Body", 8); canvas.setFillColor(colors.HexColor("#495465"))
        canvas.drawString(50, 30, "ITI | COCO - CLIP - FAISS | Phase 1 | 07/11/2026")
        canvas.drawRightString(A4[0]-50, 30, str(doc.page))
    OUT.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(filename), pagesize=A4, leftMargin=50, rightMargin=50,
                            topMargin=48, bottomMargin=58,
                            title="COCO phase 1 - " + b.pages[0]["title"],
                            author="Hamou Yasmine; Ouziala Ali; Lounis Aya", subject="Data exploration and pre-processing")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (OUT / (filename.stem + ".md")).write_text("\n".join(md), encoding="utf-8")
    from pypdf import PdfReader
    reader = PdfReader(str(filename))
    assert len(reader.pages) == len(b.pages), f"Unexpected overflow: {filename}: {len(reader.pages)} pages, expected {len(b.pages)}"
    texts = [page.extract_text() or "" for page in reader.pages]
    for title, text in zip([page["title"] for page in b.pages], texts):
        assert " ".join(title.split()) in " ".join(text.split()), f"Section/page mismatch: {title}"
    assert all("\ufffd" not in text and "\u25a0" not in text for text in texts)
    return {"file": str(filename.relative_to(ROOT)), "pages": len(reader.pages), "sha256": file_hash(filename)}


def main():
    review = read_json(REVIEW)
    # Refuse publication against a different set of evidence.
    for name, sha in review["artifact_sha256"].items():
        assert file_hash(ROOT / name) == sha, f"Review is stale: {name}; rerun scripts/review_phase1.py"
    make_figures()
    from scripts.report_pedagogy import EVIDENCE, make_pedagogical_evidence
    make_pedagogical_evidence()
    results = [draw_book(content(lang)) for lang in ("fr", "en")]
    write_json(OUT / "report_build_manifest.json", {
        "evidence_review_sha256": file_hash(REVIEW), "reports": results,
        "pedagogical_example_sha256": file_hash(EVIDENCE),
        "authors": ["Hamou Yasmine", "Ouziala Ali", "Lounis Aya"],
        "numeric_evidence": "Real local COCO artifacts; no data imported from the example report",
        "scope": "Phase 1: 10000 KB, 100 validation, 1000 test, nested original 100; no final RAG evaluation",
    })
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
