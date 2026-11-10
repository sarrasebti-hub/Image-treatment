"""Regenerate the two thin, initially unexecuted phase-1 notebooks."""
from pathlib import Path

import nbformat as nbf


def make(name, intro, code_cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"display_name": "Python (imp)", "language": "python", "name": "imp"}
    book.cells = [nbf.v4.new_markdown_cell(intro)]
    book.cells += [nbf.v4.new_code_cell(code) for code in code_cells]
    nbf.write(book, Path(__file__).parent / name)


setup = """from pathlib import Path
import sys
root = Path.cwd()
if root.name == 'notebooks':
    root = root.parent
if not (root / 'configs/config.yaml').is_file():
    raise RuntimeError('Open this notebook from the project root or notebooks directory')
sys.path.insert(0, str(root))
from src.utils.core import load_config, path, read_json
cfg = load_config(root / 'configs/config.yaml')
"""


if __name__ == "__main__":
    make("01_dataset_exploration.ipynb", """# COCO 2017 — data exploration

Phase 1, due 4 November 2026. Run data audit and metadata preparation first, then download the NLTK tagger. This notebook has no stored COCO results. Analysis of test references is descriptive only; it must not be used to tune preprocessing or construct retrieval queries.

All computation lives in `src/`. Attribute/context keywords and box relations are derived proxies, not COCO ground truth. See the README for definitions and limitations.""", [setup,
         """from src.data.coco import CocoData
from src.exploration.eda import run_eda
data = CocoData(cfg)
result = run_eda(cfg, data)
{split: {key: stats[key] for key in ('images', 'captions', 'vocabulary_size', 'objects_per_image', 'caption_length_words')}
 for split, stats in result['splits'].items()}""",
         """from IPython.display import Image, display
for figure in result['splits']['knowledge_base']['figures']:
    display(Image(filename=str(path(cfg, 'figures') / figure)))""",
         """from src.utils.report import phase_report
phase_report(cfg)"""])
    make("02_preprocessing_validation.ipynb", """# Preprocessing and artifact validation

Run split creation, image acquisition, metadata, embeddings and indexing first. All critical failures must be resolved before freezing the release. This notebook performs no final caption generation or caption-quality evaluation.

Teacher confirmation of the common test IDs and test size remains required. A successful technical QC does not make a provisional split official.""", [setup,
         """split_manifest = read_json(path(cfg, 'splits') / 'split_manifest.json')
split_manifest""",
         """from src.data.coco import CocoData
from src.preprocessing.quality import quality_control
data = CocoData(cfg)
quality_control(cfg, data)""",
         """from src.indexing.faiss_index import smoke_test
image_hits = smoke_test(cfg, data, split='validation', kind='image')
image_hits""",
         """text_hits = smoke_test(cfg, data, split='validation', kind='text')
text_hits""",
         """from src.utils.report import phase_report
phase_report(cfg)"""])
