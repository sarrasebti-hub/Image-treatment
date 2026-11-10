from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import heapq
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports"
SEED = 20261106
ImageFile.LOAD_TRUNCATED_IMAGES = False


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_path(split, filename):
    base = RAW / split
    p = base / filename
    if p.exists():
        return p
    # Archives are commonly extracted with a redundant split directory.
    return base / split / filename


def annotations(split):
    inst = read_json(RAW / "annotations" / f"instances_{split}.json")
    caps = read_json(RAW / "annotations" / f"captions_{split}.json")
    images = {int(x["id"]): x for x in inst["images"]}
    cats = {int(x["id"]): x["name"] for x in inst["categories"]}
    objs = collections.defaultdict(list)
    for a in inst["annotations"]:
        objs[int(a["image_id"])].append(a)
    capmap = collections.defaultdict(list)
    for a in caps["annotations"]:
        capmap[int(a["image_id"])].append(a)
    return inst, caps, images, cats, objs, capmap


def explore():
    t0 = time.time()
    out = OUT / "statistical_analysis"
    out.mkdir(parents=True, exist_ok=True)
    report = {"seed": SEED, "splits": {}, "methodology": {"scene_thresholds": "simple <= Q1 nonzero instance count; medium > Q1 and <= Q3; complex > Q3; empty-annotation images retained as unknown", "interaction": "COCO has no relation labels in instances; co-occurrence of person plus another category is only a candidate proxy, not verified interaction", "context": "Indoor/outdoor/day/night are not encoded in these annotations and are reported unknown"}}
    catcounts = collections.Counter()
    allcaps = []
    cooccur = np.zeros((91, 91), dtype=np.int64)
    for split in ("train2017", "val2017"):
        inst, caps, imgs, cats, objs, capmap = annotations(split)
        objcounts = np.array([len(objs.get(i, [])) for i in imgs], dtype=np.int32)
        distinct = np.array([len({int(a["category_id"]) for a in objs.get(i, [])}) for i in imgs], dtype=np.int16)
        nonzero = objcounts[objcounts > 0]
        q1, q3 = (np.quantile(nonzero, [0.25, 0.75]).tolist() if len(nonzero) else (0, 0))
        split_catcounts = collections.Counter()
        for iid, aa in objs.items():
            ids = sorted({int(a["category_id"]) for a in aa})
            catcounts.update(int(a["category_id"]) for a in aa)
            split_catcounts.update(int(a["category_id"]) for a in aa)
            if ids:
                ix = np.array([int(x)-1 for x in ids])
                cooccur[np.ix_(ix, ix)] += 1
        for a in caps["annotations"]:
            allcaps.append(a["caption"])
        invalid_associations = sum(1 for a in caps["annotations"] if int(a["image_id"]) not in imgs)
        dimrows = [{"image_id": i, "width": int(m["width"]), "height": int(m["height"]), "pixels": int(m["width"])*int(m["height"])} for i,m in imgs.items()]
        pd.DataFrame(dimrows).to_csv(out / f"{split}_dimensions.csv", index=False)
        hist = collections.Counter(map(int, objcounts))
        complexity = {"simple": int(np.sum((objcounts > 0) & (objcounts <= q1))), "medium": int(np.sum((objcounts > q1) & (objcounts <= q3))), "complex": int(np.sum(objcounts > q3)), "unknown_no_instances": int(np.sum(objcounts == 0))}
        report["splits"][split] = {"images_in_annotations": len(imgs), "object_annotations": len(inst["annotations"]), "categories": len(cats), "captions": len(caps["annotations"]), "images_without_instances": int(np.sum(objcounts==0)), "images_without_captions": sum(not capmap.get(i) for i in imgs), "captions_with_unknown_image": invalid_associations, "caption_counts": dict(collections.Counter(map(len, capmap.values()))), "instance_count_mean": float(objcounts.mean()), "instance_count_median": float(np.median(objcounts)), "instance_count_min": int(objcounts.min()), "instance_count_max": int(objcounts.max()), "multi_instance_images": int(np.sum(objcounts > 1)), "distinct_categories_mean": float(distinct.mean()), "complexity_thresholds_nonzero_q1_q3": [float(q1), float(q3)], "complexity_distribution": complexity, "dimensions_summary": {dim: {stat: float(pd.Series([r[dim] for r in dimrows]).describe()[stat]) for stat in ("mean", "min", "max")} for dim in ("width", "height", "pixels")}, "category_names": cats}
        pd.DataFrame([{"category_id": k, "category": cats[k], "instances": split_catcounts[k], "percent_of_instances": 100*split_catcounts[k]/max(1,len(inst["annotations"]))} for k in cats]).sort_values("instances", ascending=False).to_csv(out / f"{split}_category_frequencies.csv", index=False)
        pd.DataFrame([{"object_count": k, "images": v} for k,v in sorted(hist.items())]).to_csv(out / f"{split}_object_count_distribution.csv", index=False)
        print(f"Explored {split}: {len(imgs)} images, {len(inst['annotations'])} instances, {len(caps['annotations'])} captions", flush=True)
    # Captions across both official splits; preserve originals in all output data.
    tokens = [re.findall(r"[\w']+", s.casefold()) for s in allcaps]
    lengths = [len(x) for x in tokens]
    wordfreq = collections.Counter(w for row in tokens for w in row)
    pd.DataFrame(wordfreq.most_common(500), columns=["word","frequency"]).to_csv(out / "caption_word_frequencies_top500.csv", index=False)
    pd.DataFrame(collections.Counter(lengths).items(), columns=["word_count","captions"]).sort_values("word_count").to_csv(out / "caption_length_distribution.csv", index=False)
    report["captions_all_splits"] = {"count": len(allcaps), "word_count_mean": float(np.mean(lengths)), "word_count_median": float(np.median(lengths)), "word_count_min": int(min(lengths)), "word_count_max": int(max(lengths)), "vocabulary_size_casefolded_word_tokens": len(wordfreq), "top_30_words": wordfreq.most_common(30), "pos_distribution": "not computed: no installed spaCy model or NLTK tagger is assumed; run optional NLP setup described in README"}
    report["global_category_frequency"] = "Category CSVs are split-specific; top-50 train categories saved in train2017_category_frequencies.csv"
    report["cooccurrence_note"] = "Object co-occurrence is a proxy. Each matrix cell counts images in which both category IDs occur; diagonals count category presence."
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        viz=ROOT/"visualizations"; viz.mkdir(parents=True,exist_ok=True)
        cats=report["splits"]["train2017"]["category_names"]
        cdf=pd.read_csv(out/"train2017_category_frequencies.csv").head(50)
        fig,ax=plt.subplots(figsize=(12,10)); sns.barplot(data=cdf,y="category",x="instances",color="#3569a8",ax=ax); ax.set(title="Top 50 COCO object categories — train2017",xlabel="Annotated instances",ylabel="Category"); fig.tight_layout(); fig.savefig(viz/"top50_object_classes.png",dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5)); ax.hist(lengths,bins=range(0,max(lengths)+2),color="#4477aa"); ax.set(title="COCO caption length distribution",xlabel="Words per caption",ylabel="Captions"); fig.tight_layout(); fig.savefig(viz/"caption_length_distribution.png",dpi=180); plt.close(fig)
        wf=pd.DataFrame(wordfreq.most_common(30),columns=["word","frequency"]); fig,ax=plt.subplots(figsize=(11,7)); sns.barplot(data=wf,y="word",x="frequency",color="#66a182",ax=ax); ax.set(title="Most frequent caption words",xlabel="Token frequency",ylabel="Word"); fig.tight_layout(); fig.savefig(viz/"caption_word_frequencies.png",dpi=180); plt.close(fig)
        order=list(cats); mat=cooccur[np.ix_(order,order)]; pd.DataFrame(mat,index=[cats[x] for x in order],columns=[cats[x] for x in order]).to_csv(out/"train_val_category_cooccurrence.csv")
        fig,ax=plt.subplots(figsize=(20,18)); sns.heatmap(np.log1p(mat),xticklabels=[cats[x] for x in order],yticklabels=[cats[x] for x in order],cmap="mako",ax=ax); ax.set(title="Category co-occurrence by image (log1p count)",xlabel="Category",ylabel="Category"); plt.xticks(rotation=90); fig.tight_layout(); fig.savefig(viz/"object_cooccurrence_matrix.png",dpi=160); plt.close(fig)
        report["visualizations"]="Top 50 category, caption length, word frequency, and annotation-derived co-occurrence figures generated. Scene context and color are not inferred from annotation labels."
    except ImportError as e:
        report["visualizations_error"]=str(e)
    report["elapsed_seconds"] = round(time.time()-t0, 2)
    dump_json(out / "dataset_audit.json", report)
    return report


def features(imgs, cats, objs):
    rows = {}
    for iid, im in imgs.items():
        aa = objs.get(iid, [])
        ids = sorted({int(a["category_id"]) for a in aa})
        counts = collections.Counter(int(a["category_id"]) for a in aa)
        rows[iid] = {"image_id": iid, "object_count": len(aa), "category_count": len(ids), "category_ids": ids, "category_names": [cats[x] for x in ids], "counts": counts, "width": im["width"], "height": im["height"], "person_cooccurrence_proxy": 1 in ids and len(ids)>1, "annotations": aa}
    nonzero = [r["object_count"] for r in rows.values() if r["object_count"]]
    q1, q3 = np.quantile(nonzero, [0.25, 0.75])
    for r in rows.values():
        n = r["object_count"]
        r["complexity"] = "unknown" if n == 0 else ("simple" if n <= q1 else "medium" if n <= q3 else "complex")
    return rows, [float(q1), float(q3)]


def stratified_select(rows, n, seed, reserved=frozenset()):
    """Deterministic iterative rarity-first multilabel greedy coverage selector."""
    rng = np.random.default_rng(seed)
    candidates = {i:r for i,r in rows.items() if i not in reserved}
    if len(candidates) < n:
        raise RuntimeError(f"Only {len(candidates)} eligible images; need {n}")
    labels = {}
    for i,r in candidates.items():
        labs = {f"cat:{c}" for c in r["category_ids"]}
        labs.add(f"scene:{r['complexity']}")
        labs.add("multiple" if r["object_count"] > 1 else "single")
        if r["person_cooccurrence_proxy"]: labs.add("person_cooccurrence_proxy")
        labels[i] = labs
    freq = collections.Counter(x for labs in labels.values() for x in labs)
    tie = {i:float(rng.random()) for i in candidates}
    selected=[]; covered=collections.Counter(); remaining=set(candidates)
    # Rare labels guide early picks. Once marginal coverage falls, repeat labels
    # are allowed to fill the quota while retaining a stable seeded tie-break.
    def score(i):
        return sum(1/math.sqrt(freq[x])/(1+covered[x]) for x in labels[i]) + 0.00001*tie[i]
    heap=[(-score(i),i) for i in remaining]; heapq.heapify(heap)
    for _ in range(n):
        while True:
            neg_old,best=heapq.heappop(heap)
            current=score(best)
            # Submodular gains only decrease as labels become covered; old heap
            # priorities are therefore valid upper bounds for lazy greedy.
            if not heap or current >= -heap[0][0]: break
            heapq.heappush(heap,(-current,best))
        selected.append(best); remaining.remove(best); covered.update(labels[best])
    return selected, dict(covered)


def save_split(name, ids, root=ROOT):
    p=root/"data"/"selected"; p.mkdir(parents=True,exist_ok=True)
    dump_json(p/f"{name}_ids.json", {"version":"1.0.0","seed":SEED,"split":name,"image_ids":ids})
    pd.DataFrame({"image_id":ids}).to_csv(p/f"{name}_ids.csv",index=False)


def select():
    _,_,vimgs,vcats,vobjs,vcaps=annotations("val2017")
    _,_,timgs,tcats,tobjs,tcaps=annotations("train2017")
    vf,vq=features(vimgs,vcats,vobjs)
    vf={i:r for i,r in vf.items() if len(vcaps.get(i,[]))==5}
    test, testcov=stratified_select(vf,1000,SEED)
    val,valcov=stratified_select(vf,100,SEED+1,frozenset(test))
    tf,_=features(timgs,tcats,tobjs)
    tf={i:r for i,r in tf.items() if len(tcaps.get(i,[]))==5}
    train,traincov=stratified_select(tf,10000,SEED+2)
    for name,ids in (("train",train),("validation",val),("test",test)): save_split(name,ids)
    dump_json(ROOT/"data"/"selected"/"selection_report.json",{"seed":SEED,"method":"Deterministic rarity-first iterative multilabel greedy selection. Eligible images must have exactly five linked captions. Labels include object category presence, annotation-derived complexity, single/multiple instance, and person-plus-other-category co-occurrence proxy. Indoor/outdoor context is unknown: COCO instance annotations provide no scene context label. No direct interaction claims.","scene_thresholds_val2017_q1_q3":vq,"eligible_images_exactly_five_captions":{"train":len(tf),"val":len(vf)},"excluded_wrong_caption_cardinality":{"train":len(timgs)-len(tf),"val":len(vimgs)-len(vf)},"counts":{"train":len(train),"validation":len(val),"test":len(test)},"coverage_counts":{"train":traincov,"validation":valcov,"test":testcov},"validation_test_overlap":len(set(val)&set(test)),"train_source":"train2017 only","validation_test_source":"val2017","image_ids":{"train":train,"validation":val,"test":test}})
    # Image-level metadata with original boxes, category names and object counts.
    md=ROOT/"data"/"metadata"; md.mkdir(parents=True,exist_ok=True)
    for name,ids,fs,cs,os,feature_rows in (("train",train,timgs,tcats,tobjs,tf),("validation",val,vimgs,vcats,vobjs,vf),("test",test,vimgs,vcats,vobjs,vf)):
        arr=[]
        for iid in ids:
            r=feature_rows[iid]
            objects=[{"name":cs[int(a["category_id"])],"bbox":a["bbox"]} for a in r["annotations"]]
            arr.append({"image_id":iid,"filename":fs[iid]["file_name"],"split":name,"width":r["width"],"height":r["height"],"objects":objects,"object_count":len(objects),"categories":list(dict.fromkeys(obj["name"] for obj in objects))})
        dump_json(md/f"{name}.json",arr)
        pd.DataFrame([{k:v for k,v in x.items() if k!="objects"} for x in arr]).to_csv(md/f"{name}.csv",index=False)
    # Preserve raw caption string; cleaned field is explicitly separate.
    capdir=ROOT/"data"/"captions"; capdir.mkdir(parents=True,exist_ok=True)
    for name,ids,cm in (("train",train,tcaps),("validation",val,vcaps),("test",test,vcaps)):
        records=[]
        for iid in ids:
            for a in cm.get(iid,[]):
                raw=a["caption"]
                records.append({"image_id":iid,"caption_id":int(a["id"]),"caption_raw":raw,"caption_cleaned":re.sub(r"\s+"," ",raw).strip().casefold()})
        dump_json(capdir/f"{name}.json",records)
        pd.DataFrame(records).to_csv(capdir/f"{name}.csv",index=False,quoting=csv.QUOTE_MINIMAL)
    # Verify selected image files, decode, caption cardinality and exact hash dupes.
    audit={"seed":SEED,"splits":{},"hash_duplicate_groups":[],"perceptual_hash":"not run (optional imagehash dependency); exact SHA-256 computed"}
    hash_seen={}
    for name,ids,fs,cm,sp in (("train",train,timgs,tcaps,"train2017"),("validation",val,vimgs,vcaps,"val2017"),("test",test,vimgs,vcaps,"val2017")):
        reasons=[]; ncap=collections.Counter(); okay=0
        for iid in ids:
            ncap[iid]=len(cm.get(iid,[]))
            path=image_path(sp,fs[iid]["file_name"])
            try:
                with Image.open(path) as im: im.verify()
                with Image.open(path) as im: im.load()
                okay+=1
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                if sha in hash_seen: audit["hash_duplicate_groups"].append({"image_id":iid,"duplicate_of":hash_seen[sha],"sha256":sha})
                else: hash_seen[sha]=iid
            except Exception as e: reasons.append({"image_id":iid,"reason":str(e)})
        audit["splits"][name]={"expected_count":len(ids),"decoded_count":okay,"files_missing_or_corrupt":reasons,"images_with_five_captions":sum(x==5 for x in ncap.values()),"images_with_wrong_caption_count":{str(k):v for k,v in collections.Counter(ncap.values()).items() if k!=5},"all_five_captions":all(v==5 for v in ncap.values())}
    audit["overlap"]={"train_validation":len(set(train)&set(val)),"train_test":len(set(train)&set(test)),"validation_test":len(set(val)&set(test))}
    dump_json(OUT/"dataset_audit"/"selected_images_integrity.json",audit)
    # Produce selected-set comparison statistics from the actual frozen IDs.
    (OUT/"selection_report").mkdir(parents=True,exist_ok=True)
    distro=[]
    for nm,ids,rows in (("train",train,tf),("validation",val,vf),("test",test,vf)):
        cc=collections.Counter(c for iid in ids for c in rows[iid]["category_ids"])
        for cid,count in cc.items(): distro.append({"split":nm,"category_id":cid,"category":(tcats if nm=="train" else vcats)[cid],"images_present":count,"percent_images":100*count/len(ids)})
    pd.DataFrame(distro).to_csv(OUT/"selection_report"/"category_distribution.csv",index=False)
    comp=[]
    for nm,ids,rows in (("train",train,tf),("validation",val,vf),("test",test,vf)):
        for level,count in collections.Counter(rows[i]["complexity"] for i in ids).items(): comp.append({"split":nm,"complexity":level,"images":count})
    pd.DataFrame(comp).to_csv(OUT/"selection_report"/"complexity_distribution.csv",index=False)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        viz=ROOT/"visualizations"; viz.mkdir(parents=True,exist_ok=True)
        sdf=pd.DataFrame(comp)
        fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(data=sdf,x="split",y="images",hue="complexity",ax=ax); ax.set(title="Scene complexity in selected splits",xlabel="Split",ylabel="Images"); fig.tight_layout(); fig.savefig(viz/"selected_scene_complexity.png",dpi=180); plt.close(fig)
        td=pd.DataFrame([x for x in distro if x["split"]=="test"]).sort_values("images_present",ascending=False).head(30)
        fig,ax=plt.subplots(figsize=(10,8)); sns.barplot(data=td,y="category",x="images_present",color="#3569a8",ax=ax); ax.set(title="Top categories in stratified test set",xlabel="Test images containing category",ylabel="Category"); fig.tight_layout(); fig.savefig(viz/"test_category_distribution.png",dpi=180); plt.close(fig)
    except ImportError: pass
    return audit


def preprocess():
    cfg=read_json(ROOT/"configs"/"config.json")
    audit={"splits":{}}
    for name,sp in (("train","train2017"),("validation","val2017"),("test","val2017")):
        manifest=read_json(ROOT/"data"/"selected"/f"{name}_ids.json")
        _,_,imgs,_,_,_=annotations(sp)
        dest=ROOT/"data"/"processed"/name; dest.mkdir(parents=True,exist_ok=True)
        errors=[]
        for j,iid in enumerate(manifest["image_ids"],1):
            p=image_path(sp,imgs[iid]["file_name"])
            try:
                with Image.open(p) as im:
                    im=ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((cfg["image_size"],cfg["image_size"]),Image.Resampling.LANCZOS)
                    canvas=Image.new("RGB",(cfg["image_size"],cfg["image_size"]),tuple(cfg["padding_rgb"]))
                    canvas.paste(im,((cfg["image_size"]-im.width)//2,(cfg["image_size"]-im.height)//2))
                    canvas.save(dest/f"{iid:012d}.jpg",quality=95)
            except Exception as e: errors.append({"image_id":iid,"reason":str(e)})
            if j%1000==0: print(f"Processed {name}: {j}/{len(manifest['image_ids'])}",flush=True)
        audit["splits"][name]={"requested":len(manifest["image_ids"]),"written":len(manifest["image_ids"])-len(errors),"errors":errors}
    dump_json(OUT/"dataset_audit"/"preprocessing.json",audit)
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["explore","select","preprocess","all","embeddings","text_embeddings","retrieval","caption_index","phash","pos","visual","quality","extended_analysis","metadata_redesign","replacement_plan","replacement_apply","remaining"])
    args=p.parse_args()
    if args.stage == "extended_analysis":
        result=__import__("src.extended_analysis",fromlist=["run"]).run(ROOT)
        print(json.dumps(result,indent=2))
        return
    if args.stage == "metadata_redesign":
        result=__import__("src.metadata_redesign",fromlist=["run"]).run(ROOT)
        print(json.dumps(result,indent=2))
        return
    if args.stage in ("embeddings","text_embeddings","retrieval","caption_index","phash","pos","visual","quality","replacement_plan","replacement_apply","remaining"):
        cfg=read_json(ROOT/"configs"/"config.json")
        stages={
            "embeddings":lambda: __import__("src.embeddings",fromlist=["extract_all"]).extract_all(ROOT,cfg),
            "text_embeddings":lambda: __import__("src.text_embeddings",fromlist=["extract_train"]).extract_train(ROOT,cfg),
            "retrieval":lambda: __import__("src.retrieval",fromlist=["run"]).run(ROOT,cfg),
            "caption_index":lambda: __import__("src.text_embeddings",fromlist=["build_caption_index"]).build_caption_index(ROOT,cfg),
            "phash":lambda: __import__("src.perceptual_hash",fromlist=["run"]).run(ROOT,cfg),
            "pos":lambda: __import__("src.caption_pos",fromlist=["run"]).run(ROOT,cfg),
            "visual":lambda: __import__("src.visual_analysis",fromlist=["run"]).run(ROOT,cfg),
            "quality":lambda: __import__("src.quality_control",fromlist=["run"]).run(ROOT,cfg),
            "replacement_plan":lambda: __import__("src.controlled_replacement",fromlist=["plan_replacements"]).plan_replacements(ROOT,cfg),
            "replacement_apply":lambda: __import__("src.controlled_replacement",fromlist=["apply_replacements"]).apply_replacements(ROOT,cfg),
        }
        sequence=["embeddings","text_embeddings","retrieval","caption_index","phash","pos","visual","quality"] if args.stage=="remaining" else [args.stage]
        stage_dir=ROOT/"reports"/"stages"; stage_dir.mkdir(parents=True,exist_ok=True)
        for stage in sequence:
            print(f"Starting stage: {stage}",flush=True)
            try:
                result=stages[stage]()
                (stage_dir/f"{stage}.json").write_text(json.dumps({"stage":stage,"status":"PASS","result":result},indent=2,default=str)+"\n",encoding="utf-8")
                print(f"Completed stage: {stage}",flush=True)
            except Exception as exc:
                (stage_dir/f"{stage}.json").write_text(json.dumps({"stage":stage,"status":"FAIL","error":str(exc)},indent=2)+"\n",encoding="utf-8")
                raise
        return
    if args.stage in ("explore","all"): explore()
    if args.stage in ("select","all"): select()
    if args.stage in ("preprocess","all"): preprocess()
