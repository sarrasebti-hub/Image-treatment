"""Explanatory report additions, based only on saved, real COCO artifacts."""
import json
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from src.preprocessing.metadata import load_rgb
from src.utils.core import file_hash, load_config, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
EVIDENCE = ROOT / "reports/pedagogical_example.json"


def make_pedagogical_evidence():
    """Replay an existing query; do not train a model or invent new measurements."""
    cfg = load_config()
    result_path = ROOT / "reports/retrieval/validation/results.json"
    results = read_json(result_path)
    query_rows = read_json(ROOT / "reports/retrieval/validation/query_rows.json")
    qi = next(i for i, row in enumerate(query_rows) if row["image_id"] == 490)
    query = next(q for q in results["queries"] if q["image_id"] == 490)
    vectors = np.load(ROOT / "reports/retrieval/validation/query_embeddings.npy")
    vector = np.ascontiguousarray(vectors[qi:qi + 1], dtype=np.float32)
    kb_vectors = np.load(ROOT / "artifacts/embeddings/image.npy", mmap_mode="r")
    rows = read_json(ROOT / "artifacts/indexes/image_rows.json")
    index = faiss.read_index(str(ROOT / "artifacts/indexes/image.index"))
    scores, positions = index.search(vector, 5)
    metadata = {}
    for split in ("knowledge_base", "validation"):
        with (ROOT / f"artifacts/metadata/{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                metadata[row["image_id"]] = row
    assert 490 not in {r["image_id"] for r in rows}
    assert vector.shape == (1, 512) and np.isfinite(vector).all()
    assert np.isclose(np.linalg.norm(vector), 1, atol=1e-6)
    checked_hits = []
    for j, hit in enumerate(query["results"]):
        row = int(positions[0, j])
        assert row == hit["row"] and rows[row]["image_id"] == hit["image_id"]
        assert np.isclose(scores[0, j], hit["cosine_similarity"], atol=1e-6)
        dot = float(kb_vectors[row] @ vector[0])
        assert np.isclose(dot, scores[0, j], atol=1e-6)
        assert metadata[hit["image_id"]]["captions"] == hit["captions"]
        checked_hits.append({"rank": j + 1, "faiss_row_zero_based": row,
                             "image_id": hit["image_id"], "cosine_similarity": float(scores[0, j]),
                             "direct_dot_product": dot, "captions": hit["captions"]})
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg["embedding"]["model"], revision=cfg["embedding"]["revision"],
        cache_dir=str(ROOT / "data/interim/cache/huggingface"), local_files_only=True,
        trust_remote_code=False)
    image_path = ROOT / "data/raw/train2017/000000000490.jpg"
    original_hash = file_hash(image_path)
    assert original_hash == query["file_sha256"]
    rgb = load_rgb(image_path)
    pixels = processor(images=[rgb], return_tensors="np")["pixel_values"]
    assert pixels.shape == (1, 3, 224, 224) and np.isfinite(pixels).all()
    ip = processor.image_processor
    visible = np.clip(pixels[0].transpose(1, 2, 0) * np.array(ip.image_std) + np.array(ip.image_mean), 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].imshow(rgb); axes[0].set_title("COCO 490: original 500 x 495")
    axes[1].imshow(visible); axes[1].set_title("Actual CLIP input: 224 x 224\nDisplay view, denormalised")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "phase1_example_490.png", dpi=180); plt.close(fig)
    rgb.close()
    assert file_hash(image_path) == original_hash
    good = next(q for q in results["queries"] if q["image_id"] == 135250)
    assert good["results"][0]["image_id"] == 227125
    pairs = [(135250, 227125), (490, 267664)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    for axrow, (query_id, hit_id) in zip(axes, pairs):
        for ax, image_id, label in zip(axrow, (query_id, hit_id), ("Validation query", "FAISS rank 1")):
            path = ROOT / "data/raw/train2017" / f"{image_id:012d}.jpg"
            with Image.open(path) as im: ax.imshow(im)
            ax.set_title(f"{label} - COCO {image_id}", fontsize=11)
            ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "phase1_qualitative_cases.png", dpi=180); plt.close(fig)
    paths = ["reports/retrieval/validation/results.json", "reports/retrieval/validation/query_embeddings.npy",
             "reports/retrieval/validation/query_rows.json", "artifacts/embeddings/image.npy",
             "artifacts/indexes/image.index", "artifacts/indexes/image_rows.json",
             "artifacts/metadata/knowledge_base.jsonl", "artifacts/metadata/validation.jsonl"]
    write_json(EVIDENCE, {
        "scope": "Report-only trace of existing real retrieval; no new model inference or final evaluation",
        "query_image_id": 490, "query_split": "validation", "image_path": image_path.relative_to(ROOT).as_posix(),
        "file_sha256": original_hash, "original_size": [500, 495],
        "processor_tensor_shape": list(pixels.shape), "processor_dtype": str(pixels.dtype),
        "query_vector_source": "Saved vector from the executed 100-query validation run",
        "query_vector_row_zero_based": qi, "query_vector_first_six": vector[0, :6].tolist(),
        "query_vector_norm": float(np.linalg.norm(vector)), "query_not_in_image_index": True,
        "query_metadata": metadata[490], "verified_top5": checked_hits,
        "qualitative_examples": [
            {"query_image_id": 135250, "top1_image_id": 227125,
             "cosine_similarity": good["results"][0]["cosine_similarity"],
             "interpretation": "Both depict traffic lights; different viewpoint and surroundings"},
            {"query_image_id": 490, "top1_image_id": 267664,
             "cosine_similarity": checked_hits[0]["cosine_similarity"],
             "interpretation": "Query dog/skateboard; top1 cat/room. Object mismatch despite visual similarity"}],
        "qualitative_policy": "Two illustrative cases, not a representative sample or a measured relevance rate; references consulted only after retrieval",
        "artifact_sha256": {p: file_hash(ROOT / p) for p in paths}})
    for lang in ("fr", "en"): make_pipeline(lang)


def make_pipeline(lang):
    tr = lambda fr, en: fr if lang == "fr" else en
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set(xlim=(0, 12), ylim=(0, 8)); ax.axis("off")
    def box(x, y, w, h, label, color="#eaf0f5"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=.08",
                                   facecolor=color, edgecolor="#42627c", linewidth=1.1))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=13.0, color="#182f43")
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "->", "color": "#42627c", "lw": 1.3})
    box(.2, 6.95, 11.6, .75, tr("COCO : photos + captions humaines + annotations d'objets\nIDs gelés : 10 000 base + 100 validation + 1 000 test (dont les 100 initiaux)",
        "COCO: photos + human captions + object annotations\nFrozen IDs: 10,000 KB + 100 validation + 1,000 test (including the original 100)"))
    arrow(6, 6.95, 6, 6.55)
    box(.2, 5.55, 11.6, 1, tr("AUDIT : fichiers lisibles, doublons, captions, séparation des jeux\nPRÉPARATION : images RGB en mémoire / copie du texte aux espaces nettoyés\nMÉTADONNÉES + EXPLORATION : objets, boîtes, provenance, statistiques avant/après",
        "AUDIT: readable files, duplicates, captions, split separation\nPREPARATION: RGB images in memory / whitespace-cleaned text copy\nMETADATA + EXPLORATION: objects, boxes, provenance, before/after statistics"))
    for x in (2, 6, 10): arrow(x, 5.55, x, 5.05)
    box(.2, 3.75, 3.6, 1.3, tr("BASE SEULE\n10 000 images -> CLIP image\n50 000 captions -> CLIP texte\nVecteurs float32, L2, 512 nombres",
        "KNOWLEDGE BASE ONLY\n10,000 images -> CLIP image\n50,000 captions -> CLIP text\nfloat32, L2, 512 values"))
    box(4.2, 3.75, 3.6, 1.3, tr("VALIDATION : 100 photos\nMême préparation + CLIP image\n1 vecteur par requête\nSans ses captions de référence",
        "VALIDATION: 100 photos\nSame preparation + CLIP image\n1 vector per query\nNo query reference captions"), "#e7f1eb")
    box(8.2, 3.75, 3.6, 1.3, tr("TEST : 1 000 photos\n100 anciens IDs conservés\nMétadonnées + statistiques\nRéservé à la suite du projet",
        "TEST: 1,000 photos\nOriginal 100 IDs preserved\nMetadata + statistics\nReserved for the next phase"), "#fff3dc")
    arrow(2, 3.75, 2, 3.25)
    box(.2, 2.15, 3.6, 1.1, tr("FAISS IndexFlatIP\nIndex image : 10 000 vecteurs\nIndex texte : 50 000 vecteurs\nTable ligne -> image_id",
        "FAISS IndexFlatIP\nImage index: 10,000 vectors\nText index: 50,000 vectors\nRow -> image_id mapping"))
    arrow(6, 3.75, 6, 2.95); arrow(3.8, 2.7, 4.2, 2.7)
    box(4.2, 1.85, 3.6, 1.1, tr("RECHERCHE IMAGE -> IMAGE\nRequête x index image\nTri des scores cosinus\nTop-k lignes -> IDs -> captions",
        "IMAGE -> IMAGE SEARCH\nQuery x image index\nSort cosine scores\nTop-k rows -> IDs -> captions"), "#e7f1eb")
    arrow(6, 1.85, 6, 1.35)
    box(.2, .35, 7.6, 1, tr("PREUVES : JSON / CSV / galerie + contrôles qualité\n100 requêtes réelles, 500 voisins vérifiés (top-k = 5)\nIndex texte préparé, non utilisé par cette démonstration",
        "EVIDENCE: JSON / CSV / gallery + quality checks\n100 real queries, 500 checked neighbours (top-k = 5)\nText index prepared, not used by this demonstration"))
    box(8.2, .35, 3.6, 2.9, tr("SÉPARATION À RESPECTER\n\nValidation et test\nn'alimentent aucun index.\n\nIci : aucune caption générée,\naucun entraînement,\naucune évaluation finale.",
        "SPLIT BOUNDARY\n\nValidation and test\nnever enter either index.\n\nHere: no generated captions,\nno training,\nno final evaluation."), "#fff3dc")
    fig.tight_layout(pad=.4)
    fig.savefig(FIG / f"phase1_pipeline_{lang}.png", dpi=190); plt.close(fig)


def enhance_book(b):
    """Keep the existing report, inserting four explanatory pages (20 in total)."""
    t = b.tr
    e = read_json(EVIDENCE)
    for name, sha in e["artifact_sha256"].items():
        assert file_hash(ROOT / name) == sha, f"Stale worked example: {name}"
    original = list(b.pages)

    b.page("Vue d'ensemble : du fichier à la recherche", "Overview: from file to retrieval")
    b.p("Le projet prépare une bibliothèque d'exemples. Une nouvelle photo sera comparée à cette bibliothèque : on cherche des images proches, puis on récupère les phrases déjà associées à ces images. Dans cette phase, le programme ne rédige aucune nouvelle phrase.",
        "The project prepares a library of examples. A query photograph is compared with that library: similar images are found, then the sentences already attached to those images are retrieved. In this phase the program does not write a new sentence.")
    b.figure(f"phase1_pipeline_{b.lang}.png", 355,
        "Pipeline réellement exécuté. Les flèches montrent le parcours des données ; la branche test reste séparée de la recherche.",
        "Executed pipeline. Arrows show the data flow; the test branch stays separate from retrieval.")
    b.h("Lire le schéma en trois questions", "Three questions for reading the diagram")
    b.p("1. Que prépare-t-on ? Les fichiers COCO, leurs objets et leurs cinq captions humaines. L'audit vérifie les données avant de calculer les statistiques et les vecteurs. Les originaux sont conservés.",
        "1. What is prepared? COCO files, their objects and their five human captions. The audit checks the data before statistics and vectors are computed. Originals are retained.")
    b.p("2. Dans quoi cherche-t-on ? Uniquement dans les 10 000 images de la base. CLIP traduit chaque image en 512 nombres ; FAISS compare ces listes de nombres. Le fichier de correspondance permet ensuite de retrouver la bonne photo et ses captions.",
        "2. What is searched? Only the 10,000 KB images. CLIP translates each image into 512 numbers; FAISS compares those number lists. A mapping file then leads back to the correct photograph and its captions.")
    b.p("3. Comment le vérifie-t-on ? Les 100 images de validation servent de requêtes, sans leurs captions. Les 1 000 images de test sont préparées et décrites, mais leur évaluation finale attend la phase suivante.",
        "3. How is it checked? The 100 validation images act as queries without their captions. The 1,000 test images are prepared and described, but their final evaluation belongs to the next phase.")
    pipeline = b.pages[-1]

    b.page("Pourquoi ces choix ?", "Why these choices?")
    b.p("Nous avons choisi une chaîne simple à vérifier sur ce PC. Chaque choix répond à un besoin précis ; aucun test présenté ici ne démontre que cette configuration est la meilleure possible.",
        "We chose a pipeline that can be checked on this PC. Each choice serves a specific need; none of the checks here proves this is the best possible configuration.")
    b.table([t("Choix", "Choice"), t("Pourquoi ?", "Reason"), t("Limite à connaître", "Limitation")], [
        [t("IDs fixes, graine 42", "Fixed IDs, seed 42"), t("Retrouver exactement les mêmes images ; conserver le premier test de 100.", "Recover exactly the same images; preserve the original test100."), t("Un tirage local ne remplace pas la liste commune de l'enseignante.", "A local sample cannot replace the teacher's common list.")],
        [t("Objets des annotations COCO", "Objects from COCO annotations"), t("Relier directement chaque catégorie et boîte à son image, avec une source connue.", "Link each category and box to its image with a known source."), t("Ce n'est pas un détecteur exécuté sur les pixels. La boîte invalide est signalée.", "This is not a detector run on pixels. The invalid box is flagged.")],
        [t("Nettoyage léger des captions", "Light caption cleaning"), t("Corriger les espaces sans changer le sens ; conserver caption_raw pour vérifier.", "Fix whitespace without changing meaning; retain caption_raw for checking."), t("Les éventuelles erreurs de langage humaines ne sont pas corrigées.", "Possible human language errors are not corrected.")],
        [t("Processeur officiel de CLIP", "Official CLIP processor"), t("Donner au modèle le format attendu : RGB, 224 x 224 et normalisation identique.", "Give the model its expected input: RGB, 224 x 224 and matching normalisation."), t("Le recadrage central peut supprimer un objet sur le bord.", "Centre cropping can remove a peripheral object.")],
        [t("CLIP préentraîné et figé", "Frozen pretrained CLIP"), t("Représenter images et textes sans entraîner un modèle sur ce petit jeu.", "Represent images and text without training a model on this small set."), t("Une ressemblance globale ne garantit pas les bons objets ou la bonne action.", "Global similarity does not guarantee matching objects or actions.")],
        ["L2 + IndexFlatIP", t("La norme vaut 1 ; le produit scalaire devient un cosinus. Comparaison exhaustive des 10 000 images.", "Unit norm makes inner product equal cosine. Exhaustive comparison of 10,000 images."), t("Simple à contrôler, mais le coût augmente avec la taille de la base.", "Easy to check, but cost grows with KB size.")],
        ["JSONL + image_rows.json", t("Séparer les vecteurs des textes tout en gardant un lien explicite par image_id.", "Separate vectors and text while preserving an explicit image_id link."), t("Un décalage de lignes donnerait de mauvaises captions ; le contrôle vérifie ce lien.", "A row mismatch would retrieve wrong captions; checks validate this link.")]])
    b.p("Pourquoi ne pas chercher directement dans les phrases de validation ? Parce qu'on disposerait alors d'une description de la réponse. La démonstration part uniquement des pixels de la requête. L'index texte est préparé pour la suite ; il n'intervient pas dans ce test image-vers-image.",
        "Why not search using validation sentences? That would supply a description of the answer. The demonstration starts only from query pixels. The text index is prepared for later work and is not used by this image-to-image check.")
    choices = b.pages[-1]

    b.page("Exemple pas à pas : l'image COCO 490", "Worked example: COCO image 490")
    b.figure("phase1_example_490.png", 138,
        "À gauche, le JPEG conservé ; à droite, une vue de l'entrée réelle du processeur. La normalisation est inversée uniquement pour rendre cette illustration visible.",
        "Left: the retained JPEG. Right: a view of the actual processor input. Normalisation is reversed only to display this illustration.")
    b.h("1. Partir d'une photo identifiable", "1. Start from an identifiable photograph")
    b.p("Le fichier data/raw/train2017/000000000490.jpg mesure 500 x 495 pixels. Son ID COCO est 490 : il appartient aux 100 images de validation et n'apparaît pas dans l'index. Les annotations décrivent un chien et un skateboard ; elles servent ici à expliquer la scène, pas à créer la requête.",
        "data/raw/train2017/000000000490.jpg is 500 x 495 pixels. COCO ID 490 belongs to the 100-image validation set and is absent from the index. Its annotations describe a dog and a skateboard; they explain the scene here but are not used to create the query.")
    b.h("2. Passer des pixels à une liste de nombres", "2. Turn pixels into a list of numbers")
    values = ", ".join(f"{v:.6f}" for v in e["query_vector_first_six"])
    b.p("Après conversion RGB, redimensionnement, recadrage et normalisation, le tenseur a la forme 1 x 3 x 224 x 224 : une image, trois canaux de couleur. CLIP le transforme en 512 nombres. Le vecteur est divisé par sa norme pour obtenir une norme de 1.",
        "After RGB conversion, resizing, cropping and normalisation, the tensor has shape 1 x 3 x 224 x 224: one image, three colour channels. CLIP turns it into 512 values. Dividing the vector by its norm gives a unit-length vector.")
    b.code("q[:6] = [" + values + "]\nshape(q) = (512,)    norm(q) = 1.000000")
    b.h("3. Comparer, trier, retrouver l'image", "3. Compare, sort, recover the image")
    b.p("Pour chaque vecteur de la base, FAISS multiplie les nombres correspondants et additionne les 512 produits : score = somme(q[j] x v[j]). Les cinq scores les plus grands sont retenus. Le premier vaut 0,818381 et pointe vers la ligne 4450, comptée à partir de zéro.",
        "For each KB vector, FAISS multiplies corresponding values and adds the 512 products: score = sum(q[j] x v[j]). The five highest scores are retained. The first is 0.818381 and points to zero-based row 4450.")
    b.code("image_rows[4450]['image_id'] = 267664\nFAISS score = 0.818381071\nDirect dot product = 0.818381071")
    b.h("4. Relier le résultat aux captions, puis le vérifier", "4. Link the result to captions, then check it")
    b.p("L'ID 267664 mène à sa ligne de métadonnées et à ses cinq captions. L'une dit : « A lone cat sits in a dark room with some furniture. » Ce texte appartient bien au voisin : il ne décrit pas automatiquement notre chien. La recherche fonctionne, mais ce premier voisin constitue une erreur de catégorie. La confiance 1 de cette référence humaine ne rend pas le score FAISS certain.",
        "ID 267664 leads to its metadata row and five captions. One says: 'A lone cat sits in a dark room with some furniture.' It correctly belongs to the neighbour, but does not automatically describe our dog. Retrieval works, yet this first neighbour has the wrong animal category. The human-reference confidence of 1 does not make the FAISS score certain.")
    b.p("Preuve reproductible : reports/pedagogical_example.json contient le vecteur sauvegardé (six premières valeurs), sa norme et le recalcul des cinq scores. La galerie et les cinq voisins complets figurent dans reports/retrieval/manual_490_k5/.",
        "Reproducible evidence: reports/pedagogical_example.json records the saved vector's first six values, its norm and the five recomputed scores. The gallery and full five neighbours are under reports/retrieval/manual_490_k5/.")
    worked = b.pages[-1]

    b.page("Comprendre les réussites et les erreurs", "Understanding successes and errors")
    b.figure("phase1_qualitative_cases.png", 225,
        "Deux exemples des résultats déjà sauvegardés. Ligne du haut : feux de circulation. Ligne du bas : le chien de la requête est remplacé par un chat au premier rang.",
        "Two examples from already saved results. Top row: traffic lights. Bottom row: the query dog is replaced by a cat at rank one.")
    b.h("Ce qui réussit : la chaîne est vérifiable", "What succeeds: the pipeline can be checked")
    b.p("Les 11 100 fichiers sont lisibles, les jeux sont séparés et chaque vecteur mène à la bonne image et à ses vraies captions. Les 500 voisins des 100 requêtes ont été rejoués. C'est une réussite de préparation et de traçabilité ; cela ne mesure pas encore la qualité d'un futur générateur de captions.",
        "All 11,100 files are readable, splits are separate and each vector leads to the correct image and its real captions. All 500 neighbours from 100 queries were replayed. This establishes preparation and traceability, not the quality of a future caption generator.")
    b.h("Ce qui réussit parfois : retrouver le même type de contenu", "What sometimes succeeds: retrieving the same type of content")
    b.p("Pour la requête 135250, le premier voisin 227125 montre également un feu de circulation (score 0,838155). L'objet principal concorde, mais le cadrage et l'environnement diffèrent. Ce cas illustre une recherche utile sans prouver que toutes les requêtes sont aussi bonnes.",
        "For query 135250, first neighbour 227125 also shows a traffic light (score 0.838155). The main object matches, but viewpoint and surroundings differ. This illustrates useful retrieval without proving all queries are equally good.")
    b.h("Ce qui échoue : ressemblance ne veut pas dire même scène", "What fails: similar appearance is not the same scene")
    b.p("Pour 490, le premier voisin contient un chat au lieu d'un chien. Les voisins de rangs 3 et 4 montrent un chien et un skateboard : ils reprennent davantage les objets de la requête, sans garantir la même action. La pièce sombre et les formes générales peuvent expliquer le premier résultat ; c'est une hypothèse de lecture visuelle, pas une explication mesurée du modèle.",
        "For 490, the first neighbour contains a cat instead of a dog. Ranks three and four show a dog and skateboard, sharing more query objects without guaranteeing the same action. The dark room and overall shapes may explain the first result; this is a visual interpretation, not a measured model explanation.")
    b.p("Ces deux cas ont été choisis pour expliquer le fonctionnement. Ils ne constituent ni un taux de réussite ni une évaluation finale. Les références de validation ont été consultées après la recherche pour commenter les images. Les limites connues restent documentées : boîte invalide isolée, doublons textuels, objets hors recadrage et attributs heuristiques. Les corriger ou comparer plusieurs méthodes demandera un protocole séparé.",
        "These two cases were chosen to explain behaviour. They are neither a success rate nor a final evaluation. Validation references were consulted after retrieval to discuss the pictures. Known limits remain documented: one invalid box, shared caption strings, objects outside the crop and heuristic attributes. Addressing them or comparing methods requires a separate protocol.")
    discussion = b.pages[-1]

    b.pages = original[:2] + [pipeline, choices] + original[2:13] + [worked, discussion] + original[13:]
    assert len(b.pages) == 20
    for section, page in enumerate(b.pages[1:], 1):
        import re
        page["title"] = f"{section}. " + re.sub(r"^\d+\.\s*", "", page["title"])
    toc = [
        [t("Vue d'ensemble", "Overview"), t("Quel parcours suit une image ?", "What path does an image follow?"), "3"],
        [t("Choix et données", "Choices and data"), t("Pourquoi ces outils et ces jeux ?", "Why these tools and splits?"), "4-5"],
        [t("Audit et préparation", "Audit and preparation"), t("Que vérifie-t-on et que transforme-t-on ?", "What is checked and transformed?"), "6-9"],
        [t("Exploration", "Exploration"), t("Que montrent les statistiques ?", "What do the statistics show?"), "10-13"],
        ["CLIP / FAISS", t("Comment la recherche fonctionne-t-elle ?", "How does retrieval work?"), "14-15"],
        [t("Exemple et discussion", "Example and discussion"), t("Suivre une photo et comprendre les erreurs", "Trace a photograph and understand errors"), "16-17"],
        [t("Reproduction", "Reproduction"), t("Quelles commandes relancer ?", "Which commands should be run?"), "18-19"],
        [t("Bilan et limites", "Status and limitations"), t("Ce qui est prêt et ce qui reste à valider", "What is ready and what needs approval"), "20"]]
    for j, (kind, value) in enumerate(b.pages[1]["blocks"]):
        if kind == "table":
            b.pages[1]["blocks"][j] = ("table", (value[0], toc))
    return b
