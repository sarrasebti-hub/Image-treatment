"""Plan and apply four audited one-for-one test replacements without touching train/validation."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .pipeline import ROOT, RAW, annotations, dump_json, features, image_path
from .perceptual_hash import phash_image

OLD_TEST_IDS=(525083,133819,129062,465129)
REPLACEMENT_SEED=20261107
MANIFEST_FILES=(
    "data/selected/train_ids.json","data/selected/train_ids.csv",
    "data/selected/validation_ids.json","data/selected/validation_ids.csv",
    "data/selected/test_ids.json","data/selected/test_ids.csv",
    "data/selected/selection_report.json",
)
AUDIT_BACKUP_FILES=(
    "reports/perceptual_hash/duplicate_candidates.json",
    "reports/perceptual_hash/hash_statistics.json",
    "reports/perceptual_hash/phash_database.npz",
    "reports/dataset_audit/selected_images_integrity.json",
    "reports/perceptual_hash/cross_split_visual_review.json",
    "reports/perceptual_hash/cross_split_pair_evidence.json",
    "reports/perceptual_hash/cross_split_candidates.png",
    "reports/FINAL_SCIENTIFIC_AUDIT.md",
)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _source_image(split,iid):
    name=f"{int(iid):012d}.jpg"
    p=RAW/split/split/name
    if not p.exists(): p=RAW/split/name
    return p


def _processed_candidate_hash(path,config):
    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        im=ImageOps.exif_transpose(source).convert("RGB")
        im.thumbnail((int(config["image_size"]),int(config["image_size"])),Image.Resampling.LANCZOS)
        canvas=Image.new("RGB",(int(config["image_size"]),int(config["image_size"])),tuple(config["padding_rgb"]))
        canvas.paste(im,((canvas.width-im.width)//2,(canvas.height-im.height)//2))
        b=io.BytesIO(); canvas.save(b,format="JPEG",quality=95); b.seek(0)
        with Image.open(b) as encoded:
            return phash_image(encoded.convert("RGB"))


def _labels(row):
    labs={f"cat:{c}" for c in row["category_ids"]}
    labs.add(f"scene:{row['complexity']}")
    labs.add("multiple" if row["object_count"]>1 else "single")
    if row["person_cooccurrence_proxy"]: labs.add("person_cooccurrence_proxy")
    return labs


def _hamming_min(hashes,reference,block_size):
    result=np.empty(len(hashes),dtype=np.uint8)
    for start in range(0,len(hashes),block_size):
        end=min(start+block_size,len(hashes))
        xor=np.bitwise_xor(hashes[start:end,None],reference[None,:])
        counts=np.unpackbits(xor.view(np.uint8).reshape(end-start,len(reference),8),axis=2).sum(axis=2)
        result[start:end]=counts.min(axis=1).astype(np.uint8)
    return result


def plan_replacements(root:Path=ROOT,config=None):
    config=config or _read(root/"configs/config.json")
    train=_read(root/"data/selected/train_ids.json")["image_ids"]
    val=_read(root/"data/selected/validation_ids.json")["image_ids"]
    test_doc=_read(root/"data/selected/test_ids.json"); test=list(map(int,test_doc["image_ids"]))
    selection=_read(root/"data/selected/selection_report.json")
    prior_hashes={rel:_sha(root/rel) for rel in MANIFEST_FILES}
    if not all(i in test for i in OLD_TEST_IDS):
        raise RuntimeError("Controlled replacement targets do not all occur in the current test manifest")
    inst,caps,imgs,cats,objs,capmap=annotations("val2017")
    val_features,quartiles=features(imgs,cats,objs)
    used=set(map(int,train))|set(map(int,val))|set(test)
    eligible={iid:r for iid,r in val_features.items() if iid not in used and len(capmap.get(iid,[]))==5}
    if not eligible:
        raise RuntimeError("No eligible val2017 candidates with exactly five captions remain")
    # Existing train/validation pHashes are authoritative and are read from the
    # original pHash database; original train/validation manifests stay frozen.
    z=np.load(root/"reports/perceptual_hash/phash_database.npz")
    splits=z["splits"].astype(str); ids=z["image_ids"].astype(np.int64); stored=z["hashes"].astype(np.uint64)
    base_mask=np.isin(splits,["train","validation"])
    base_hashes=stored[base_mask]
    if len(base_hashes)!=len(train)+len(val):
        raise RuntimeError(f"Expected pHashes for {len(train)+len(val)} train/validation images, found {len(base_hashes)}")
    threshold=int(config["perceptual_hash"]["hamming_distance_threshold"])
    # Exact file hashes for the current selected dataset (excluding all four
    # soon-to-be-replaced test rows) prevent candidates duplicating any survivor.
    selected_sha={}
    for split,id_list in (("train2017",train),("val2017",val),("val2017",[i for i in test if i not in OLD_TEST_IDS])):
        for iid in id_list:
            p=_source_image(split,int(iid))
            if not p.is_file(): raise FileNotFoundError(f"Selected source image missing: {p}")
            digest=_sha(p)
            selected_sha.setdefault(digest,[]).append({"split":split,"image_id":int(iid)})
    eligible_ids=sorted(eligible)
    candidate_hashes=[]; candidate_sha=[]; decode_errors=[]
    for j,iid in enumerate(eligible_ids,1):
        p=_source_image("val2017",iid)
        try:
            candidate_hashes.append(_processed_candidate_hash(p,config)); candidate_sha.append(_sha(p))
        except Exception as exc:
            candidate_hashes.append(np.uint64(0)); candidate_sha.append("")
            decode_errors.append({"image_id":iid,"reason":str(exc)})
        if j%500==0: print(f"Screened replacement candidates: {j}/{len(eligible_ids)}",flush=True)
    hamming=_hamming_min(np.asarray(candidate_hashes,dtype=np.uint64),base_hashes,int(config["perceptual_hash"].get("block_size",128)))
    error_ids={x["image_id"] for x in decode_errors}
    safe={iid:eligible[iid] for pos,iid in enumerate(eligible_ids)
          if iid not in error_ids and hamming[pos]>threshold and candidate_sha[pos] not in selected_sha}
    safe_hash={iid:candidate_sha[eligible_ids.index(iid)] for iid in safe}
    # We replace one-for-one within the same annotation-derived multi-label
    # strata: complexity, instance multiplicity, category mix, and person proxy.
    target_rows={iid:val_features[iid] for iid in OLD_TEST_IDS}
    catfreq=collections.Counter(c for row in safe.values() for c in row["category_ids"])
    rng=np.random.default_rng(REPLACEMENT_SEED)
    tie={iid:float(rng.random()) for iid in safe}
    remaining=set(safe); chosen=[]; details=[]
    targets=sorted(OLD_TEST_IDS,key=lambda iid:(
        sum(1 for r in safe.values() if r["complexity"]==target_rows[iid]["complexity"]),iid))
    for old_id in targets:
        target=target_rows[old_id]; tset=set(target["category_ids"])
        def score(iid):
            cand=safe[iid]; cset=set(cand["category_ids"]); union=tset|cset
            weights={c:1/math.sqrt(max(1,catfreq[c])) for c in union}
            weighted_jaccard=sum(weights[c] for c in tset&cset)/max(1e-12,sum(weights.values()))
            complexity=2.0 if cand["complexity"]==target["complexity"] else 0.0
            multiplicity=1.0 if (cand["object_count"]>1)==(target["object_count"]>1) else 0.0
            person=0.5 if cand["person_cooccurrence_proxy"]==target["person_cooccurrence_proxy"] else 0.0
            count_penalty=.025*abs(cand["object_count"]-target["object_count"])
            rarity=sum(1/math.sqrt(max(1,catfreq[c])) for c in cset-tset)*.025
            return 5*weighted_jaccard+complexity+multiplicity+person+rarity-count_penalty+tie[iid]*1e-5
        if not remaining:
            raise RuntimeError(f"Replacement pool exhausted after {len(chosen)} selections")
        new_id=max(remaining,key=score); remaining.remove(new_id); chosen.append(new_id)
        target_labels=sorted(_labels(target)); new_labels=sorted(_labels(safe[new_id]))
        pos=eligible_ids.index(new_id)
        details.append({"old_test_image_id":old_id,"new_test_image_id":new_id,"reason":"replaced as a train/test pHash candidate from the prior review",
            "selection_rationale":"same-stratum replacement ranked by rarity-weighted COCO category overlap, scene-complexity match, single/multiple-instance match, person-co-occurrence proxy match, and object-count proximity; deterministic tie-break seed",
            "old_complexity":target["complexity"],"new_complexity":safe[new_id]["complexity"],"old_object_count":target["object_count"],"new_object_count":safe[new_id]["object_count"],
            "old_category_names":target["category_names"],"new_category_names":safe[new_id]["category_names"],"old_labels":target_labels,"new_labels":new_labels,
            "valid_caption_count":len(capmap[new_id]),"candidate_min_hamming_to_train_validation":int(hamming[pos]),"hamming_threshold":threshold,
            "candidate_sha256":safe_hash[new_id],"candidate_file":_source_image("val2017",new_id).relative_to(root).as_posix()})
    plan={"status":"READY" if len(chosen)==4 else "FAIL","revision":"1.1.0","replacement_seed":REPLACEMENT_SEED,"selection_seed_original":int(config["seed"]),
        "hamming_threshold_inclusive":threshold,"screened_eligible_candidates":len(eligible_ids),"decode_errors":decode_errors,
        "excluded_already_selected_or_reserved":len(val_features)-len(eligible),"eligible_exactly_five_caption_candidates":len(eligible),
        "pHash_safe_candidates_after_screen":len(safe),"pHash_rejected_candidates":int(np.sum(hamming<=threshold)),"exact_sha_rejected_candidates":sum(x in selected_sha for x in candidate_sha if x),
        "pHash_reference_count_train_validation":len(base_hashes),"replacement_pairs":details,
        "manifest_sha256_before":prior_hashes,
        "method":"Screen all unused val2017 images with exactly five COCO captions. Decode originals, hash processed-equivalent 224x224 RGB JPEGs using existing 64-bit DCT pHash, require minimum Hamming distance >6 from every selected train/validation pHash, and reject any original-file SHA-256 matching a surviving selected image. Choose four deterministic one-for-one replacements matching complexity/multiplicity strata and maximizing rarity-weighted category overlap with removed IDs.",
        "source_annotations_unchanged":True}
    if len(details)!=4: plan["status"]="FAIL"
    dest=root/"reports/selection_report/replacement_candidate_screening.json"
    dump_json(dest,plan)
    return plan


def _atomic_text(path:Path,text:str):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(text,encoding="utf-8",newline="\n")
    os.replace(tmp,path)


def _caption_record(iid,ann):
    raw=ann["caption"]
    import re
    return {"image_id":int(iid),"caption_id":int(ann["id"]),"caption_raw":raw,"caption_cleaned":re.sub(r"\s+"," ",raw).strip().casefold()}


def _metadata_record(iid,source,im,cats,aa,row):
    objects=[{"name":cats[int(a["category_id"])],"bbox":a["bbox"]} for a in aa]
    split="train" if source=="train2017" else "test"
    return {"image_id":int(iid),"filename":f"{int(iid):012d}.jpg","split":split,"width":int(im["width"]),"height":int(im["height"]),
        "objects":objects,"object_count":len(objects),"categories":list(dict.fromkeys(obj["name"] for obj in objects))}


def apply_replacements(root:Path=ROOT,config=None):
    config=config or _read(root/"configs/config.json")
    plan=_read(root/"reports/selection_report/replacement_candidate_screening.json")
    if plan.get("status")!="READY" or len(plan.get("replacement_pairs",[]))!=4:
        raise RuntimeError("Replacement plan is not READY with exactly four candidates")
    current={rel:_sha(root/rel) for rel in MANIFEST_FILES}
    if current!=plan["manifest_sha256_before"]:
        raise RuntimeError("Frozen manifests changed after candidate screening; refusing to apply stale replacement plan")
    backup=root/"data/selected/backups/phase1_pre_test_replacement_v1"
    backup.mkdir(parents=True,exist_ok=False)
    backup_rows=[]
    for rel in MANIFEST_FILES+AUDIT_BACKUP_FILES:
        source=root/rel; target=backup/rel
        if not source.is_file(): raise FileNotFoundError(f"Required backup source missing: {source}")
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        backup_rows.append({"path":rel,"backup_path":target.relative_to(root).as_posix(),"sha256":_sha(source),"backup_sha256":_sha(target)})
    if any(x["sha256"]!=x["backup_sha256"] for x in backup_rows): raise RuntimeError("Backup SHA-256 verification failed")
    dump_json(backup/"backup_inventory.json",{"revision":"1.1.0","files":backup_rows,"note":"Original manifests and pHash/SHA audit outputs are immutable historical backups; do not delete."})
    old_test=_read(root/"data/selected/test_ids.json")
    old_ids=list(map(int,old_test["image_ids"])); test_ids=old_ids.copy()
    replace_map={int(x["old_test_image_id"]):int(x["new_test_image_id"]) for x in plan["replacement_pairs"]}
    for idx,iid in enumerate(test_ids):
        if iid in replace_map: test_ids[idx]=replace_map[iid]
    if len(test_ids)!=1000 or len(set(test_ids))!=1000: raise RuntimeError("Replacement would change test count or create duplicate IDs")
    train_ids=list(map(int,_read(root/"data/selected/train_ids.json")["image_ids"]))
    val_ids=list(map(int,_read(root/"data/selected/validation_ids.json")["image_ids"]))
    if set(test_ids)&(set(train_ids)|set(val_ids)): raise RuntimeError("Replacement IDs overlap with train or validation")
    # Build updated metadata/captions from original COCO val annotations only.
    _,_,imgs,cats,objs,capmap=annotations("val2017")
    val_features,_=features(imgs,cats,objs)
    test_caps=[]; test_meta=[]
    for iid in test_ids:
        if len(capmap.get(iid,[]))!=5: raise RuntimeError(f"Replacement/final test image {iid} does not have five captions")
        test_caps.extend(_caption_record(iid,a) for a in capmap[iid])
        test_meta.append(_metadata_record(iid,"val2017",imgs[iid],cats,objs.get(iid,[]),val_features[iid]))
    # Prepare all output content before committing the three versioned files.
    test_doc={**old_test,"version":"1.1.0","revision":"controlled_test_replacement_v1","replacement_seed":plan["replacement_seed"],"image_ids":test_ids,
        "replacements":[{"old_image_id":a,"new_image_id":b} for a,b in replace_map.items()]}
    selection=_read(root/"data/selected/selection_report.json")
    if selection["image_ids"]["test"]!=old_ids: raise RuntimeError("Selection report and test manifest diverged before apply")
    selection["image_ids"]["test"]=test_ids; selection["counts"]["test"]=len(test_ids)
    cov=collections.Counter()
    for iid in test_ids: cov.update(_labels(val_features[iid]))
    selection["coverage_counts"]["test"]=dict(cov)
    selection["validation_test_overlap"]=len(set(test_ids)&set(val_ids))
    selection["manifest_revision"]="1.1.0"
    selection["replacement_selection"]={"seed":plan["replacement_seed"],"source_split":"val2017","method":plan["method"],"hamming_threshold_inclusive":plan["hamming_threshold_inclusive"],"replacement_pairs":plan["replacement_pairs"]}
    selection["replacement_history"]={"previous_manifest_revision":"1.0.0","backup_directory":backup.relative_to(root).as_posix(),"controlled_replacement_count":4,
        "old_ids_removed":list(replace_map),"new_ids_added":list(replace_map.values()),"train_validation_manifests_modified":False}
    _atomic_text(root/"data/selected/test_ids.json",json.dumps(test_doc,indent=2)+"\n")
    _atomic_text(root/"data/selected/test_ids.csv","image_id\n"+"".join(f"{iid}\n" for iid in test_ids))
    _atomic_text(root/"data/selected/selection_report.json",json.dumps(selection,indent=2,ensure_ascii=False)+"\n")
    # Regenerate only replacement test images; old processed images remain in place.
    generated=[]
    for iid in replace_map.values():
        p=_source_image("val2017",iid); dest=root/"data/processed/test"/f"{iid:012d}.jpg"
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((config["image_size"],config["image_size"]),Image.Resampling.LANCZOS)
            canvas=Image.new("RGB",(config["image_size"],config["image_size"]),tuple(config["padding_rgb"]))
            canvas.paste(im,((canvas.width-im.width)//2,(canvas.height-im.height)//2)); canvas.save(dest,quality=95)
        generated.append({"image_id":iid,"processed_path":dest.relative_to(root).as_posix(),"sha256":_sha(dest)})
    # Replace test caption/object metadata exports using official annotations.
    dump_json(root/"data/captions/test.json",test_caps)
    pd.DataFrame(test_caps).to_csv(root/"data/captions/test.csv",index=False,quoting=csv.QUOTE_MINIMAL)
    dump_json(root/"data/metadata/test.json",test_meta)
    pd.DataFrame([{k:v for k,v in x.items() if k!="objects"} for x in test_meta]).to_csv(root/"data/metadata/test.csv",index=False)
    # Update test category/complexity summaries while retaining train/validation rows.
    dist_path=root/"reports/selection_report/category_distribution.csv"
    if dist_path.exists():
        dist=pd.read_csv(dist_path); dist=dist[dist["split"]!="test"].copy()
    else: dist=pd.DataFrame(columns=["split","category_id","category","images_present","percent_images"])
    cc=collections.Counter(c for iid in test_ids for c in val_features[iid]["category_ids"])
    new_dist=pd.DataFrame([{"split":"test","category_id":cid,"category":cats[cid],"images_present":n,"percent_images":100*n/len(test_ids)} for cid,n in cc.items()])
    pd.concat([dist,new_dist],ignore_index=True).to_csv(dist_path,index=False)
    comp_path=root/"reports/selection_report/complexity_distribution.csv"
    if comp_path.exists(): comp=pd.read_csv(comp_path); comp=comp[comp["split"]!="test"].copy()
    else: comp=pd.DataFrame(columns=["split","complexity","images"])
    new_comp=pd.DataFrame([{"split":"test","complexity":k,"images":v} for k,v in collections.Counter(val_features[i]["complexity"] for i in test_ids).items()])
    pd.concat([comp,new_comp],ignore_index=True).to_csv(comp_path,index=False)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(data=pd.concat([comp,new_comp],ignore_index=True),x="split",y="images",hue="complexity",ax=ax); ax.set(title="Scene complexity in selected splits",xlabel="Split",ylabel="Images"); fig.tight_layout(); fig.savefig(root/"visualizations/selected_scene_complexity.png",dpi=180); plt.close(fig)
        td=new_dist.sort_values("images_present",ascending=False).head(30)
        fig,ax=plt.subplots(figsize=(10,8)); sns.barplot(data=td,y="category",x="images_present",color="#3569a8",ax=ax); ax.set(title="Top categories in stratified test set",xlabel="Test images containing category",ylabel="Category"); fig.tight_layout(); fig.savefig(root/"visualizations/test_category_distribution.png",dpi=180); plt.close(fig)
    except ImportError: pass
    # Update the preprocessing report without rewriting unchanged split records.
    prep_path=root/"reports/dataset_audit/preprocessing.json"
    prep=_read(prep_path) if prep_path.exists() else {"splits":{}}
    prep.setdefault("splits",{}).setdefault("test",{"requested":1000,"written":1000,"errors":[]})["controlled_replacements"]=generated
    dump_json(prep_path,prep)
    revision_history={"revision":"1.1.0","replacement_time_local":"recorded by filesystem timestamps","backup_directory":backup.relative_to(root).as_posix(),
        "manifest_files_before_sha256":current,"manifest_files_after_sha256":{rel:_sha(root/rel) for rel in MANIFEST_FILES},"replacement_pairs":plan["replacement_pairs"],
        "new_processed_images":generated,"train_validation_ids_unchanged":True,"original_images_not_moved_or_deleted":True}
    hist_dir=root/"data/selected/replacement_history"; hist_dir.mkdir(parents=True,exist_ok=True)
    dump_json(hist_dir/"revision_1.json",revision_history)
    return revision_history
