import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ANN_DIR = Path(r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations")
OUT_DIR = Path("splits")
SEED = 42
N_TEST = 1000

rng = random.Random(SEED)

kb_ids = set(json.load(open(OUT_DIR / "kb_train_ids.json")))
val_ids = set(json.load(open(OUT_DIR / "val_ids.json")))
test100 = json.load(open(OUT_DIR / "test_ids.json"))      # the 100 you already have

# same candidate list and same 3 strata as in validation.py
vcaps = defaultdict(list)
for a in json.load(open(ANN_DIR / "captions_val2017.json"))["annotations"]:
    vcaps[a["image_id"]].append(a["caption"])

inst = json.load(open(ANN_DIR / "instances_val2017.json"))
n_objects = Counter(a["image_id"] for a in inst["annotations"])

cands = sorted(
    (i for i in vcaps if len(vcaps[i]) >= 5 and n_objects[i] >= 1),
    key=lambda i: (n_objects[i], i),
)
third = len(cands) // 3
strata = [cands[:third], cands[third:2 * third], cands[2 * third:]]
sizes = [N_TEST // 3 + (1 if k < N_TEST % 3 else 0) for k in range(3)]   # 334, 333, 333

# keep the 100 existing images, then top up each stratum to its target size
test1000 = []
for group, size in zip(strata, sizes):
    already = [i for i in test100 if i in set(group)]
    rest = [i for i in group if i not in set(already)]
    test1000 += already + rng.sample(rest, size - len(already))
test1000 = sorted(test1000)

# ---------- checks ----------
assert len(test1000) == N_TEST and len(set(test1000)) == N_TEST
assert set(test100) <= set(test1000), "the 100 must be inside the 1000"
assert not set(test1000) & kb_ids, "test overlaps with knowledge base!"
assert not set(test1000) & val_ids, "test overlaps with validation!"

json.dump(test1000, open(OUT_DIR / "test_ids_1000.json", "w"))
print("test 1000:", len(test1000), "| contains the 100:", set(test100) <= set(test1000))
print("objects per image in each stratum (min-max):",
      [(n_objects[g[0]], n_objects[g[-1]]) for g in strata])