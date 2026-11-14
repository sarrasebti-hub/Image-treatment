"""Génère un mini-COCO SYNTHÉTIQUE (même format que COCO 2017) pour tester le pipeline hors-ligne.
Usage: python tools/make_fake_coco.py --out data_fake/raw --n-train 1500 --n-val 400
"""
import argparse, json, random
from pathlib import Path
from PIL import Image, ImageDraw

CATS = """person:person bicycle:vehicle car:vehicle motorcycle:vehicle airplane:vehicle bus:vehicle train:vehicle truck:vehicle
boat:vehicle traffic_light:outdoor fire_hydrant:outdoor stop_sign:outdoor parking_meter:outdoor bench:outdoor bird:animal cat:animal
dog:animal horse:animal sheep:animal cow:animal elephant:animal bear:animal zebra:animal giraffe:animal backpack:accessory
umbrella:accessory handbag:accessory tie:accessory suitcase:accessory frisbee:sports skis:sports snowboard:sports sports_ball:sports
kite:sports baseball_bat:sports baseball_glove:sports skateboard:sports surfboard:sports tennis_racket:sports bottle:kitchen
wine_glass:kitchen cup:kitchen fork:kitchen knife:kitchen spoon:kitchen bowl:kitchen banana:food apple:food sandwich:food orange:food
broccoli:food carrot:food hot_dog:food pizza:food donut:food cake:food chair:furniture couch:furniture potted_plant:furniture
bed:furniture dining_table:furniture toilet:furniture tv:electronic laptop:electronic mouse:electronic remote:electronic
keyboard:electronic cell_phone:electronic microwave:appliance oven:appliance toaster:appliance sink:appliance refrigerator:appliance
book:indoor clock:indoor vase:indoor scissors:indoor teddy_bear:indoor hair_drier:indoor toothbrush:indoor""".split()
CATS = [(i + 1 if i < 11 else i + 2, c.split(":")[0].replace("_", " "), c.split(":")[1]) for i, c in enumerate(CATS)]
COLORS = {"red": (200, 40, 40), "blue": (40, 70, 200), "green": (40, 160, 60), "yellow": (230, 200, 40),
          "white": (235, 235, 235), "black": (25, 25, 25), "brown": (120, 80, 40), "orange": (240, 130, 30)}
SCENES = {"indoor": "in a room", "outdoor": "on a street", "mixed": "in a place"}


def gen(split, n, start_id, out: Path, rng):
    (out / "images" / f"{split}").mkdir(parents=True, exist_ok=True)
    images, anns, caps = [], [], []
    aid = cid = start_id * 20
    for k in range(n):
        iid = start_id + k
        w, h = rng.choice([(320, 240), (240, 320), (400, 300), (300, 300)])
        bright = rng.choice([0.9, 0.9, 0.9, 0.25])  # quelques images "nuit"
        bg = tuple(int(v * bright) for v in rng.choice([(180, 200, 230), (200, 190, 170), (150, 170, 150)]))
        img = Image.new("RGB", (w, h), bg); d = ImageDraw.Draw(img)
        n_obj = rng.choice([1, 2, 2, 3, 4, 5, 6, 8, 11, 14])
        cats = rng.sample(CATS, min(len(CATS), rng.choice([1, 1, 2, 3, 4])))
        words = []
        for _ in range(n_obj):
            c = rng.choice(cats); cname, col = c[1], rng.choice(list(COLORS))
            bw, bh = rng.randint(20, w // 2), rng.randint(20, h // 2)
            x, y = rng.randint(0, w - bw), rng.randint(0, h - bh)
            fill = tuple(int(v * bright) for v in COLORS[col])
            (d.ellipse if rng.random() < .5 else d.rectangle)([x, y, x + bw, y + bh], fill=fill)
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": c[0], "bbox": [x, y, bw, bh], "area": bw * bh,
                         "iscrowd": 0, "segmentation": []})
            words.append((col, cname))
        fn = f"{iid:012d}.jpg"
        im = img
        if k == 3:                      # une image niveaux de gris (test conversion RGB)
            im = img.convert("L")
        im.save(out / "images" / split / fn, quality=90)
        if k == 5:                      # une image avec EXIF orientation=6 (test correction)
            ex = Image.Exif(); ex[0x0112] = 6
            img.save(out / "images" / split / fn, quality=90, exif=ex)
        images.append({"id": iid, "file_name": fn, "width": w, "height": h, "coco_url": f"http://fake/{split}/{fn}"})
        for j in range(5 if k % 97 else 6):  # une image à 6 captions
            col, nm = rng.choice(words)
            where = SCENES[rng.choice(["indoor", "outdoor", "mixed"])]
            tmpl = rng.choice([f"A {col} {nm} {where}.", f"There is a {col} {nm} and {len(words)} objects.",
                               f"{col.capitalize()} {nm}s are shown {where}", f"A photo of a {nm} , near a {rng.choice(words)[1]}"])
            cid += 1; caps.append({"id": cid, "image_id": iid, "caption": tmpl})
    cat_json = [{"id": i, "name": n, "supercategory": s} for i, n, s in CATS]
    base = {"info": {}, "licenses": [], "images": images, "categories": cat_json}
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    json.dump({**base, "annotations": anns}, open(out / "annotations" / f"instances_{split}.json", "w"))
    json.dump({**base, "annotations": caps}, open(out / "annotations" / f"captions_{split}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_fake/raw"); ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed); out = Path(a.out)
    gen("train2017", a.n_train, 1000, out, rng); gen("val2017", a.n_val, 900000, out, rng)
    print("fake COCO ->", out)
