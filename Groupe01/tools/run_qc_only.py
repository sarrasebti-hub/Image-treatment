import contextlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

nb = json.loads((ROOT / "07_qc_leakage.ipynb").read_text(encoding="utf-8-sig"))
namespace = {"__name__": "__qc_only__"}
log_path = ROOT / "logs" / "qc.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

with log_path.open("w", encoding="utf-8") as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"sys.executable: {sys.executable}")
        for index in (1, 3, 5, 7, 9):
            cell = nb["cells"][index]
            if cell["cell_type"] == "code":
                print(f"--- Executing cell {index} ---")
                exec("".join(cell["source"]), namespace)
        print("--- Calling QC main() ---")
        namespace["main"]()
