import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "test_1000_val2017"

def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())

def split_ids(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return {int(x["image_id"]) for x in d["images"]}

def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

new = split_ids(EXP / "splits" / "test_1000.json")
kb = split_ids(ROOT / "artifacts/splits/kb.json")
val = split_ids(ROOT / "artifacts/splits/val.json")
old = split_ids(ROOT / "artifacts/splits/test.json")

kb_rows = read_jsonl(ROOT / "artifacts/tables/kb_captions.jsonl")
ref_rows = read_jsonl(EXP / "tables" / "eval_refs_test_1000.jsonl")
kb_caps = {norm(r["caption_raw"]) for r in kb_rows}
test_rows = [(int(r["image_id"]), norm(r["caption_raw"])) for r in ref_rows]
if not kb_caps or not test_rows:
    sys.exit("FAIL: empty caption set")

kb_caption_image_ids = {int(r["image_id"]) for r in kb_rows}
matches = [(i, c) for i, c in test_rows if c in kb_caps]
report = {
    "n_test_images": len(new), "n_kb_images": len(kb), "n_val_images": len(val), "n_old_test_images": len(old),
    "test_inter_kb": len(new & kb), "test_inter_val": len(new & val), "test_inter_old_test": len(new & old),
    "test_inter_kb_caption_image_ids": len(new & kb_caption_image_ids),
    "n_test_reference_rows": len(test_rows), "n_unique_kb_captions": len(kb_caps),
    "normalized_caption_matches_rows": len(matches),
    "normalized_caption_matches_images": len({i for i, _ in matches}),
    "note": "Caption matches are generic text from DIFFERENT images (not same-image leakage). Same-image leakage = test_inter_kb_caption_image_ids.",
}
(ROOT / "reports" / "leak_check_test1000.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(report, indent=2))
bad = report["test_inter_kb"] or report["test_inter_val"] or report["test_inter_old_test"] or report["test_inter_kb_caption_image_ids"]
sys.exit(1 if bad else 0)