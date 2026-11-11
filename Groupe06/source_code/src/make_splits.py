"""Step 2 - Create the frozen splits, run the leakage audits, check reproducibility.

kb (10,000) and val (100) come from ONE shuffled train2017 pool (disjoint by construction).
test (1,000) comes from val2017, stratified by number of annotated objects.
The call order of the single RNG must not change, otherwise the IDs change.

Outputs:
  splits/   kb_ids, val_ids, test_ids, *_manifest, split_meta.json, excluded_captions.json
  report/   caption_policy_stats.json, leakage_audit.json, caption_leakage_stats.json,
            audit_report.json (audit + selected_images_check)
Usage:  python src/data_audit.py && python src/make_splits.py
"""
import collections
import hashlib
import json
import os
import random
import sys

from tqdm import tqdm

import config as C


def load_captions():
    cap = {s: C.load_json(C.ANN / f"captions_{s}.json") for s in ["train2017", "val2017"]}
    id2file, id2caps, n_raw = {}, {}, {}
    for s in cap:
        id2file[s] = {i["id"]: i["file_name"] for i in cap[s]["images"]}
        tmp = collections.defaultdict(list)
        for a in sorted(cap[s]["annotations"], key=lambda a: a["id"]):
            tmp[a["image_id"]].append(a["caption"].strip())
        n_raw[s] = {k: len(v) for k, v in tmp.items()}
        id2caps[s] = {k: v[:C.N_CAPS] for k, v in tmp.items() if len(v) >= C.N_CAPS}
    return id2file, id2caps, n_raw


def pick(pool, n, split, seen, rng, img_dir, id2file):
    pool = sorted(pool)
    rng.shuffle(pool)
    chosen, rejected = [], []
    for i in tqdm(pool, desc=f"{split} n={n}"):
        if len(chosen) == n:
            break
        p = img_dir[split] / id2file[split][i]
        ok, info = C.check_image(p)
        if not ok:
            rejected.append({"id": i, "reason": "unreadable", "detail": info})
            continue
        h = C.md5(p)
        if h in seen:
            rejected.append({"id": i, "reason": "duplicate_content"})
            continue
        seen[h] = i
        chosen.append(i)
    return chosen, rejected


def manifest(ids, split, id2file, id2caps, extra=None):
    return [{"image_id": i, "file_name": id2file[split][i], "captions": id2caps[split][i],
             **({"stratum": extra[i]} if extra else {})} for i in sorted(ids)]


def caption_leak(ids, split, key, kb_ids, id2caps):
    kb_index = collections.defaultdict(set)
    for i in kb_ids:
        for c in id2caps["train2017"][i]:
            kb_index[key(c)].add(i)
    hits = [(i, c) for i in ids for c in id2caps[split][i] if key(c) in kb_index]
    return {"matching_captions": len(hits),
            "images_affected": len({i for i, _ in hits}),
            "kb_images_involved": len({k for _, c in hits for k in kb_index[key(c)]})}


def main():
    C.ensure_dirs()
    img_dir = C.img_dirs()
    id2file, id2caps, n_raw = load_captions()
    train_pool = list(id2caps["train2017"].keys())
    rng = random.Random(C.SEED)  # single RNG: call order below must not change

    # ---- splits -------------------------------------------------------------
    seen_train, seen_val = {}, {}
    chosen, rej_train = pick(train_pool, C.N_KB + C.N_VAL, "train2017", seen_train, rng, img_dir, id2file)
    kb_ids, val_ids = chosen[:C.N_KB], chosen[C.N_KB:]

    inst = C.load_json(C.ANN / "instances_val2017.json")
    cnt = collections.Counter(a["image_id"] for a in inst["annotations"])
    bins = {"simple": [], "medium": [], "complex": []}
    for i, n in cnt.items():
        if i in id2caps["val2017"]:
            bins["simple" if n <= 3 else "medium" if n <= 7 else "complex"].append(i)
    print("pool per stratum:", {k: len(v) for k, v in bins.items()})

    test_ids, rej_test, test_bin = [], [], {}
    for k, q in C.QUOTA.items():
        c, r = pick(bins[k], q, "val2017", seen_val, rng, img_dir, id2file)
        test_ids += c
        rej_test += r
        for i in c:
            test_bin[i] = k
    print(len(kb_ids), len(val_ids), len(test_ids), "| rejected:", len(rej_train), len(rej_test))

    # ---- save splits + caption policy stats ----------------------------------
    for name, ids in [("kb_ids", kb_ids), ("val_ids", val_ids), ("test_ids", test_ids)]:
        C.save_json(sorted(ids), C.OUT_SPLITS / f"{name}.json")
    C.save_json(manifest(kb_ids, "train2017", id2file, id2caps), C.OUT_SPLITS / "kb_manifest.json")
    C.save_json(manifest(val_ids, "train2017", id2file, id2caps), C.OUT_SPLITS / "val_manifest.json")
    C.save_json(manifest(test_ids, "val2017", id2file, id2caps, test_bin), C.OUT_SPLITS / "test_manifest.json")
    C.save_json({"seed": C.SEED, "quota_test": C.QUOTA, "caption_policy": "first 5 captions by annotation id",
                 "rejected_train": rej_train, "rejected_test": rej_test},
                C.OUT_SPLITS / "split_meta.json", indent=2)

    stats = {}
    for name, ids, s in [("kb", kb_ids, "train2017"), ("val", val_ids, "train2017"), ("test", test_ids, "val2017")]:
        extra = [n_raw[s][i] - C.N_CAPS for i in ids]
        stats[name] = {"images": len(ids), "images_with_more_than_5_captions": sum(e > 0 for e in extra),
                       "captions_dropped": sum(extra), "captions_kept": len(ids) * C.N_CAPS}
    C.save_json(stats, C.OUT_REPORT / "caption_policy_stats.json", indent=2)

    # ---- leakage audit: IDs and image content (MD5) --------------------------
    kb, va, te = set(kb_ids), set(val_ids), set(test_ids)
    kb_hashes = {h for h, i in seen_train.items() if i in kb}
    val_hashes = {h for h, i in seen_train.items() if i in va}
    leak = {
        "sizes": {"kb": len(kb), "val": len(va), "test": len(te)},
        "kb_val_overlap": len(kb & va),
        "kb_test_id_overlap": len(kb & te),
        "val_test_id_overlap": len(va & te),
        "kb_test_content_overlap": len(set(seen_train) & set(seen_val)),
        "val_content_in_kb": len(val_hashes & kb_hashes),
        "rejected_unreadable": sum(r["reason"] == "unreadable" for r in rej_train + rej_test),
        "rejected_duplicates": sum(r["reason"] == "duplicate_content" for r in rej_train + rej_test),
    }
    assert leak["sizes"] == {"kb": C.N_KB, "val": C.N_VAL, "test": sum(C.QUOTA.values())}
    assert leak["kb_val_overlap"] == leak["kb_test_id_overlap"] == leak["val_test_id_overlap"] == 0
    assert leak["kb_test_content_overlap"] == leak["val_content_in_kb"] == 0
    print(json.dumps(leak, indent=2))
    audit_path = C.OUT_REPORT / "audit_report.json"
    base = [r for r in C.load_json(audit_path) if "split" in r] if audit_path.exists() else []
    C.save_json(base + [{"selected_images_check": leak}], audit_path, indent=2)
    C.save_json(leak, C.OUT_REPORT / "leakage_audit.json", indent=2)

    # ---- leakage audit: captions (exact / lowercase / normalized) -------------
    leak_stats = {n: {"test_vs_kb": caption_leak(test_ids, "val2017", k, kb_ids, id2caps),
                      "val_vs_kb": caption_leak(val_ids, "train2017", k, kb_ids, id2caps)}
                  for n, k in C.LEVELS.items()}
    print(json.dumps(leak_stats, indent=2))

    # ---- exclude leaking captions from the KB side (images are kept) ---------
    norm = C.LEVELS["normalized"]
    ref = collections.defaultdict(list)
    for i in test_ids:
        for c in id2caps["val2017"][i]:
            ref[norm(c)].append(("test", i))
    for i in val_ids:
        for c in id2caps["train2017"][i]:
            ref[norm(c)].append(("val", i))
    excluded = [{"kb_image_id": i, "caption": c,
                 "matched_with": [{"split": s, "image_id": j} for s, j in ref[norm(c)]]}
                for i in sorted(kb_ids) for c in id2caps["train2017"][i] if norm(c) in ref]
    per_img = collections.Counter(e["kb_image_id"] for e in excluded)
    summary = {"excluded_captions": len(excluded),
               "kb_images_affected": len(per_img),
               "max_removed_from_one_image": max(per_img.values(), default=0),
               "kb_images_left_with_0_captions": sum(v == C.N_CAPS for v in per_img.values()),
               "kb_captions_remaining": C.N_KB * C.N_CAPS - len(excluded),
               "caption_leak_before_exclusion": leak_stats}
    print(json.dumps({k: v for k, v in summary.items() if k != "caption_leak_before_exclusion"}, indent=2))
    C.save_json(excluded, C.OUT_SPLITS / "excluded_captions.json", indent=1)
    C.save_json(summary, C.OUT_REPORT / "caption_leakage_stats.json", indent=2)

    # ---- reproducibility check against the frozen v1.0 ID files ---------------
    ok = True
    for fname, expected in C.FROZEN_SHA256.items():
        got = hashlib.sha256((C.OUT_SPLITS / fname).read_bytes()).hexdigest()
        ok &= got == expected
        print(f"{fname:15s} {'MATCH' if got == expected else 'MISMATCH'} {got[:16]}...")
    if not ok:
        print("WARNING: IDs differ from the frozen v1.0 release - do NOT overwrite the repository files.")
        sys.exit(1)
    print("Regenerated IDs are identical to the frozen v1.0 release.")


if __name__ == "__main__":
    main()
