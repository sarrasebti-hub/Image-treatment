import contextlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

verify = "--verify" in sys.argv[1:]
sys.argv = ["08_freeze.ipynb"] + (["--verify"] if verify else [])
nb = json.loads((ROOT / "08_freeze.ipynb").read_text(encoding="utf-8-sig"))
namespace = {"__name__": "__freeze_only__"}
log_path = ROOT / "logs" / "freeze.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

with log_path.open("a", encoding="utf-8") as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"sys.executable: {sys.executable}")
        for index in (1, 3, 5):
            cell = nb["cells"][index]
            if cell["cell_type"] == "code":
                print(f"--- Executing cell {index} ---")
                exec("".join(cell["source"]), namespace)
        print("--- Calling freeze main() ---")
        try:
            namespace["main"]()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
