"""Step 1 - Data audit of COCO 2017 (train2017 and val2017).

Checks: counts, duplicate image IDs, images missing on disk, orphan captions,
empty captions, images with fewer than 5 captions, captions-per-image distribution.
Output: report/audit_report.json
Usage:  python src/data_audit.py
"""
import collections
import json
import os

import config as C


def audit(split, img_dir):
    d = C.load_json(C.ANN / f"captions_{split}.json")
    imgs, anns = d["images"], d["annotations"]
    ids = [i["id"] for i in imgs]
    caps = collections.defaultdict(list)
    for a in anns:
        caps[a["image_id"]].append(a["caption"])
    on_disk = set(os.listdir(img_dir))
    return {
        "split": split,
        "num_images": len(ids),
        "num_captions": len(anns),
        "duplicate_image_ids": len(ids) - len(set(ids)),
        "images_missing_on_disk": sum(1 for i in imgs if i["file_name"] not in on_disk),
        "orphan_captions": len(set(caps) - set(ids)),
        "empty_captions": sum(1 for a in anns if not a["caption"].strip()),
        "images_with_less_than_5": sum(1 for i in ids if len(caps[i]) < 5),
        "captions_per_image_distribution": dict(collections.Counter(len(caps[i]) for i in ids)),
    }


def main():
    C.ensure_dirs()
    dirs = C.img_dirs()
    reports = [audit(s, dirs[s]) for s in ["train2017", "val2017"]]
    for r in reports:
        print(json.dumps(r, indent=2))
    C.save_json(reports, C.OUT_REPORT / "audit_report.json", indent=2)


if __name__ == "__main__":
    main()
