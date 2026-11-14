import sys
import os
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# تعيين وضع الأوففلاين لضمان عدم حدوث أي محاولات اتصال عشوائية بالهب
os.environ["HF_HUB_OFFLINE"] = "1"

print(f"sys.executable: {sys.executable}")
import torch, transformers, faiss, numpy
print(f"Versions -> torch: {torch.__version__}, transformers: {transformers.__version__}, faiss: {faiss.__version__}, numpy: {numpy.__version__}")
print(f"START_TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}")

nb_path = ROOT / "04_embeddings_index.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8-sig"))

# الخلايا المطلوبة حسب ترتيب 0-based: 2, 4, 6, 10, 12, 14, 16, 18
target_indices = [2, 4, 6, 10, 12, 14, 16, 18]
namespace = {"__name__": "__siglip_only__"}

print("Extracting and executing notebook cells for SigLIP pipeline...")
for idx in target_indices:
    cell = nb["cells"][idx]
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        print(f"--- Executing Cell {idx} ---")
        exec(code, namespace)

sys.argv = ["04_embeddings_index.ipynb", "--encoders", "siglip-b16", "--skip-sbert"]

parser = namespace["argparse"].ArgumentParser()
parser.add_argument("--encoders", nargs="*")
parser.add_argument("--skip-sbert", action="store_true")
args = parser.parse_args()
cfg = namespace["load_config"]()
encoder_names = args.encoders or list(cfg["encoders"])

print(f"ENCODERS: {encoder_names}")
if encoder_names != ["siglip-b16"]:
    raise SystemExit(f"Aborting: expected only ['siglip-b16'], got {encoder_names!r}")
if not args.skip_sbert:
    raise SystemExit("Aborting: --skip-sbert must be active")

if os.environ.get("SIGLIP_DRY_RUN") == "1":
    print("DRY RUN: stopped before main(); no model loaded or artifact files written.")
else:
    print("Executing main() once with arguments: --encoders siglip-b16 --skip-sbert")
    namespace["main"]()
    print(f"END_TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("SigLIP run pipeline completed successfully.")
