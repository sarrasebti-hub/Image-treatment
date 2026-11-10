from collections import Counter
from pathlib import Path

from pycocotools.coco import COCO

from src.utils.core import file_hash, path, read_json


def image_path(cfg, source, image):
    root = path(cfg, "train_images" if source == "train2017" else "val_images").resolve()
    filename = (root / image["file_name"]).resolve()
    if not filename.is_relative_to(root):
        raise ValueError("Image filename escapes configured image root")
    return filename


class CocoFiles:
    """Resolve local image files without loading annotation trees."""
    def __init__(self, cfg):
        self.cfg = cfg

    def image_path(self, source, image):
        return image_path(self.cfg, source, image)


class CocoData(CocoFiles):
    """Read official captions/instances together; reject malformed joins early."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.captions = {}
        self.instances = {}
        self.annotation_hashes = {}
        self.validation_errors = []
        for source in ("train2017", "val2017"):
            for kind, target in (("captions", self.captions), ("instances", self.instances)):
                filename = path(cfg, "annotations") / f"{kind}_{source}.json"
                if not filename.is_file():
                    raise FileNotFoundError(f"Missing {filename}. Run download-annotations or configure local annotations.")
                raw = read_json(filename)
                for field in ("images", "annotations"):
                    ids = [row["id"] for row in raw[field]]
                    duplicates = [i for i, n in Counter(ids).items() if n > 1]
                    if duplicates:
                        self.validation_errors.append(f"{filename.name}: duplicate {field} IDs: {duplicates[:10]}")
                image_ids = {row["id"] for row in raw["images"]}
                orphans = [a["id"] for a in raw["annotations"] if a["image_id"] not in image_ids]
                if orphans:
                    self.validation_errors.append(f"{filename.name}: orphan annotations: {orphans[:10]}")
                # Do not keep a second full instance-annotation tree while COCO rereads it.
                del raw
                target[source] = COCO(str(filename))
                self.annotation_hashes[filename.name] = file_hash(filename)
        self.categories = self.instances["train2017"].cats
        if self.categories != self.instances["val2017"].cats:
            self.validation_errors.append("Train and validation category tables differ")
        for source in self.captions:
            if set(self.captions[source].imgs) != set(self.instances[source].imgs):
                self.validation_errors.append(f"{source}: captions/instances image inventories differ")

    def require_valid(self):
        if self.validation_errors:
            raise ValueError("; ".join(self.validation_errors))

    def objects(self, source, image_id):
        return self.instances[source].imgToAnns.get(image_id, [])

    def references(self, source, image_id):
        return self.captions[source].imgToAnns.get(image_id, [])

    def eligible(self, source):
        return sorted(i for i in self.captions[source].imgs if i in self.instances[source].imgs
                      and len(self.references(source, i)) == self.cfg["expected_captions"])
