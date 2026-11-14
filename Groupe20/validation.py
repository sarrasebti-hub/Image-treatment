import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ANN_DIR = Path(r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations")
OUT_DIR = Path("splits")
SEED = 42
N_VAL = 100
N_TEST = 100   # change to 1000 if your professors say so

random.seed(SEED)

# ---------- validation: 100 train2017 images NOT in the knowledge base ----------
kb_ids = set(json.load(open(OUT_DIR / "kb_train_ids.json")))

train = json.load(open(ANN_DIR / "captions_train2017.json"))
caps = defaultdict(list)
for a in train["annotations"]:
    caps[a["image_id"]].append(a["caption"])

val_pool = sorted(i for i, c in caps.items() if len(c) >= 5 and i not in kb_ids)
val_ids = sorted(random.sample(val_pool, N_VAL))

# ---------- test: stratified sample of val2017 by number of objects ----------
vcaps = defaultdict(list)
for a in json.load(open(ANN_DIR / "captions_val2017.json"))["annotations"]:
    vcaps[a["image_id"]].append(a["caption"])

inst = json.load(open(ANN_DIR / "instances_val2017.json"))
n_objects = Counter(a["image_id"] for a in inst["annotations"])

# candidates: at least 1 object and at least 5 captions
cands = sorted(
    (i for i in vcaps if len(vcaps[i]) >= 5 and n_objects[i] >= 1),
    key=lambda i: (n_objects[i], i),
)

# split into 3 equal groups: simple / medium / complex scenes
third = len(cands) // 3
strata = [cands[:third], cands[third:2 * third], cands[2 * third:]]
sizes = [N_TEST // 3 + (1 if k < N_TEST % 3 else 0) for k in range(3)]

test_ids = []
for group, size in zip(strata, sizes):
    test_ids += random.sample(group, size)
test_ids = sorted(test_ids)

# ---------- anti-leakage check ----------
assert not set(val_ids) & kb_ids, "val overlaps with knowledge base!"
assert not set(test_ids) & kb_ids, "test overlaps with knowledge base!"
assert not set(test_ids) & set(val_ids), "test overlaps with val!"

json.dump(val_ids, open(OUT_DIR / "val_ids.json", "w"))
json.dump(test_ids, open(OUT_DIR / "test_ids.json", "w"))
print("val:", len(val_ids), "| test:", len(test_ids))
print("objects per image in each stratum (min-max):",
      [(n_objects[g[0]], n_objects[g[-1]]) for g in strata])