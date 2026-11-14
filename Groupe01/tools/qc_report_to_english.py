"""Render the frozen QC report (French labels) as an English Markdown file.

`artifacts/qc/qc_report.json` is part of the frozen release g1-v1.1 (hashed in the manifest) and is
therefore NOT modified. This script only writes a translated *view*:  reports/qc_report_g1-v1.1_en.md

    python tools/qc_report_to_english.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXACT = {
    "taille split kb": "KB split size",
    "taille split val": "validation split size",
    "taille split test": "test split size (legacy, 100 images)",
    "hash des IDs kb = splits_meta": "KB ID hash = splits_meta",
    "hash des IDs val = splits_meta": "validation ID hash = splits_meta",
    "hash des IDs test = splits_meta": "test ID hash = splits_meta",
    "splits deux à deux disjoints": "splits pairwise disjoint",
    "images invalides (corrompues/absentes)": "invalid images (corrupt/missing)",
    "IDs uniques dans images.csv": "unique IDs in images.csv",
    "doublons exacts (sha256) inter/intra splits": "exact duplicates (sha256) across/within splits",
    "pas de NaN dans les stats pixel": "no NaN in pixel statistics",
    "KB : 5 captions / image": "KB: 5 captions per image",
    "val : 5 références / image": "validation: 5 references per image",
    "test : 5 références / image": "test: 5 references per image",
    "caption_id uniques (KB)": "unique caption_id (KB)",
    "ANTI-FUITE : aucune image val/test dans kb_captions": "ANTI-LEAK: no val/test image in kb_captions",
    "ANTI-FUITE : aucun caption_id d'éval dans la KB": "ANTI-LEAK: no evaluation caption_id in the KB",
    "captions brutes inchangées (raw != clean autorisé, raw non vide)": "raw captions unchanged (raw != clean allowed, raw non-empty)",
    "info : refs val identiques à une caption KB (texte)": "info: validation references textually identical to a KB caption",
    "info : refs test identiques à une caption KB (texte)": "info: test references textually identical to a KB caption",
    "métadonnées pour toutes les images": "metadata for all images",
    "attributs texte uniquement pour la KB": "text attributes for the KB only",
    "ANTI-FUITE : val jamais retrouvé pour lui-même (k=10)": "ANTI-LEAK: validation image never retrieved as itself (k=10)",
    "ANTI-FUITE : test jamais retrouvé pour lui-même (k=10)": "ANTI-LEAK: test image never retrieved as itself (k=10)",
    "info : val avec near-duplicate dans la KB": "info: validation images with a near-duplicate in the KB",
    "info : test avec near-duplicate dans la KB": "info: test images with a near-duplicate in the KB",
    "fumée val : NN-caption CIDEr > aléatoire": "smoke test val: nearest-neighbour caption CIDEr > random",
    "fumée test : NN-caption CIDEr > aléatoire": "smoke test test: nearest-neighbour caption CIDEr > random",
}
PATTERNS = [
    (r"^index (\S+) : ntotal=KB et auto-retrouvabilité$", r"index \1: ntotal = KB size and self-retrievability"),
]
DETAIL = [
    ("normal pour des captions génériques; pas une fuite d'image", "normal for generic captions; not an image leak"),
    ("aléatoire", "random"),
]


def tr_check(s: str) -> str:
    if s in EXACT:
        return EXACT[s]
    for pat, rep in PATTERNS:
        if re.match(pat, s):
            return re.sub(pat, rep, s)
    return s


def tr_detail(s: str) -> str:
    for fr, en in DETAIL:
        s = s.replace(fr, en)
    return s


def main() -> int:
    q = json.loads((ROOT / "artifacts/qc/qc_report.json").read_text(encoding="utf-8"))
    checks = q["checks"]
    untranslated = [c["check"] for c in checks if tr_check(c["check"]) == c["check"] and re.search(r"[àâéèêîôùç]|\b(pas|aucun|taille)\b", c["check"])]
    if untranslated:
        print("Untranslated check names:", untranslated, file=sys.stderr)
        return 1
    n_pass = sum(c["status"] == "PASS" for c in checks)
    lines = [f"# QC — {q['version']} (English view)", "",
             f"**{'PASS' if q['passed'] else 'FAIL'}** — {len(checks)} checks, {q['n_fail']} failure(s), {q['n_warn']} warning(s).",
             "",
             "_Translated view of the frozen `artifacts/qc/qc_report.json` (French labels), which is hashed in the release manifest and left unchanged._",
             "", "| # | Status | Check | Detail |", "|---:|---|---|---|"]
    for i, c in enumerate(checks, 1):
        lines.append(f"| {i} | {c['status']} | {tr_check(c['check'])} | {tr_detail(c.get('detail', '')).replace('|', '/')} |")
    out = ROOT / "reports" / "qc_report_g1-v1.1_en.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)} ({n_pass}/{len(checks)} PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
