"""CLIP zero-shot scene/context predictions plus separate COCO and pixel metadata."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .embeddings import combined_index


def _softmax(scores):
    x=scores-scores.max(axis=1,keepdims=True)
    e=np.exp(x)
    return e/e.sum(axis=1,keepdims=True)


def _color_stats(path):
    from PIL import Image
    rgb=np.asarray(Image.open(path).convert("RGB").resize((96,96)),dtype=np.float32)/255
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    mx=rgb.max(2); mn=rgb.min(2); delta=mx-mn
    hue=np.zeros_like(mx)
    mask=delta>1e-6
    mr=(mx==r)&mask; mg=(mx==g)&mask; mb=(mx==b)&mask
    hue[mr]=((g[mr]-b[mr])/delta[mr])%6
    hue[mg]=(b[mg]-r[mg])/delta[mg]+2
    hue[mb]=(r[mb]-g[mb])/delta[mb]+4
    hue*=60
    sat=np.where(mx>0,delta/np.maximum(mx,1e-6),0)
    bins=np.floor(hue/60).astype(int)%6
    chroma=bins[sat>.18]
    labels=["red","yellow","green","cyan","blue","magenta"]
    if chroma.size:
        counts=np.bincount(chroma,minlength=6); idx=int(counts.argmax()); color=labels[idx]; conf=float(counts[idx]/chroma.size)
    else: color="low-saturation/neutral"; conf=float(1-sat.mean())
    lum=.2126*r+.7152*g+.0722*b
    return {"dominant_hue_proxy":color,"hue_pixel_share_confidence":conf,"mean_luminance":float(lum.mean()),"mean_saturation":float(sat.mean()),"method":"coarse pixel histogram; not semantic object-color recognition"}


def run(root: Path, config: dict):
    cfg=config["visual_analysis"]; model_name=cfg.get("model",config["embedding_models"]["image"])
    output=root/"data"/"metadata"/"visual_predictions.jsonl"
    meta_path=root/"data"/"metadata"/"visual_predictions.metadata.json"
    image_matrix,rows=combined_index(root,config)
    cached={}
    if output.exists() and meta_path.exists():
        try:
            meta=json.loads(meta_path.read_text(encoding="utf-8"))
            with output.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r=json.loads(line); cached[(int(r["image_id"]),r["split"])]=r
            expected=[(int(r["image_id"]),r["split"]) for r in rows]
            if meta.get("model")==model_name and list(cached.keys())==expected:
                return json.loads((root/"reports"/"visual_analysis"/"summary.json").read_text(encoding="utf-8"))
        except (OSError,ValueError,json.JSONDecodeError): cached={}
    device_name=cfg.get("device","auto")
    device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else "cpu" if device_name=="auto" else device_name)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("Visual analysis requested CUDA but CUDA is unavailable")
    try:
        model=CLIPModel.from_pretrained(model_name,local_files_only=True).eval().to(device)
        processor=CLIPProcessor.from_pretrained(model_name,local_files_only=True)
    except Exception as exc: raise RuntimeError(f"Cannot load local CLIP model '{model_name}': {exc}") from exc
    groups={
        "context":["A photograph of an indoor scene.","A photograph of an outdoor scene."],
        "time":["A photograph taken during daytime.","A photograph taken at night."],
        "scene":[f"A photograph of {label}." for label in cfg["scene_labels"]]
    }
    prompt_vecs={}
    with torch.inference_mode():
        for group,prompts in groups.items():
            inp=processor(text=prompts,return_tensors="pt",padding=True,truncation=True).to(device)
            text_out=model.text_model(input_ids=inp["input_ids"],attention_mask=inp.get("attention_mask"))
            feat=model.text_projection(text_out.pooler_output)
            prompt_vecs[group]=torch.nn.functional.normalize(feat.float(),p=2,dim=-1).cpu().numpy()
    scale=float(model.logit_scale.exp().detach().cpu())
    metadata={}
    for split in ("train","validation","test"):
        metadata.update({int(x["image_id"]):x for x in json.loads((root/"data"/"metadata"/f"{split}.json").read_text(encoding="utf-8"))})
    threshold=float(cfg.get("confidence_threshold",.55)); scene_threshold=float(cfg.get("scene_confidence_threshold",.35))
    records=[]; uncertain=collections.Counter(); scene_counts=collections.Counter(); errors=[]
    bsize=int(cfg.get("batch_size",64))
    for start in range(0,len(rows),bsize):
        end=min(start+bsize,len(rows)); vec=np.asarray(image_matrix[start:end],dtype=np.float32)
        pred={}
        for group,txt in prompt_vecs.items():
            probs=_softmax((vec@txt.T)*scale)
            pred[group]=probs
        for off,row in enumerate(rows[start:end]):
            iid=int(row["image_id"]); ann=metadata.get(iid,{})
            old=cached.get((iid,row["split"]))
            if old is not None and old.get("model")==model_name:
                records.append(old)
                continue
            ctxp=pred["context"][off]; ci=int(ctxp.argmax()); cconf=float(ctxp[ci]); context=["indoor","outdoor"][ci] if cconf>=threshold else "UNKNOWN"; uncertain["context"]+=context=="UNKNOWN"
            tp=pred["time"][off]; ti=int(tp.argmax()); tconf=float(tp[ti]); daytime=["daytime","nighttime"][ti] if tconf>=threshold else "UNKNOWN"; uncertain["time"]+=daytime=="UNKNOWN"
            sp=pred["scene"][off]; si=int(sp.argmax()); sconf=float(sp[si]); scene=cfg["scene_labels"][si] if sconf>=scene_threshold else "UNKNOWN"; uncertain["scene"]+=scene=="UNKNOWN"; scene_counts[scene]+=1
            path=root/row["processed_path"]
            try: color=_color_stats(path)
            except Exception as exc: color={"error":str(exc)}; errors.append({"image_id":iid,"reason":str(exc)})
            records.append({"image_id":iid,"split":row["split"],"model":model_name,
                "predictions":{"indoor_outdoor":{"label":context,"confidence":cconf,"scores":{"indoor":float(ctxp[0]),"outdoor":float(ctxp[1])}},
                               "day_night":{"label":daytime,"confidence":tconf,"scores":{"daytime":float(tp[0]),"nighttime":float(tp[1])}},
                               "scene":{"label":scene,"confidence":sconf,"scores":{label:float(score) for label,score in zip(cfg["scene_labels"],sp)}}},
                "coco_annotation_objects":{"category_names":ann.get("categories",[]),"source":"COCO instance annotations; annotations, not model predictions"},
                "pixel_attributes":color,"confidence_calibration":"CLIP prompt softmax scores are relative model scores, not calibrated probabilities; labels below configured thresholds are UNKNOWN"})
        print(f"Visual predictions: {end}/{len(rows)}",flush=True)
    uncertain=collections.Counter(); scene_counts=collections.Counter()
    for r in records:
        for key in ("indoor_outdoor","day_night","scene"):
            uncertain["context" if key=="indoor_outdoor" else "time" if key=="day_night" else "scene"] += r["predictions"][key]["label"]=="UNKNOWN"
        scene_counts[r["predictions"]["scene"]["label"]]+=1
    tmp=output.with_suffix(".jsonl.tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    tmp.replace(output)
    meta_path.write_text(json.dumps({"model":model_name,"count":len(records),"image_vector_source":"cached normalized CLIP image embeddings","confidence_threshold":threshold,"scene_confidence_threshold":scene_threshold,"prompt_logit_scale":scale,"semantics":"zero-shot predictions are not ground truth"},indent=2)+"\n",encoding="utf-8")
    confidence_summary={}
    for key in ("indoor_outdoor","day_night","scene"):
        vals=[r["predictions"][key]["confidence"] for r in records]
        accepted=[v for r in records if r["predictions"][key]["label"]!="UNKNOWN" for v in [r["predictions"][key]["confidence"]]]
        confidence_summary[key]={"mean_score":float(np.mean(vals)) if vals else None,"mean_accepted_score":float(np.mean(accepted)) if accepted else None,"min_score":float(np.min(vals)) if vals else None}
    summary={"status":"PASS" if not errors and len(records)==len(rows) else "WARNING","model":model_name,"images":len(records),"confidence_summary":confidence_summary,"uncertain_counts":dict(uncertain),"scene_prediction_counts":dict(scene_counts),"color_analysis":"coarse pixel histogram and luminance; not object-level color recognition","errors":errors,"limitations":"CLIP zero-shot softmax is not calibrated. Scene labels are model predictions. Object categories are copied separately from COCO annotations; no interaction is inferred."}
    report=root/"reports"/"visual_analysis"; report.mkdir(parents=True,exist_ok=True)
    (report/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (report/"uncertain_predictions.json").write_text(json.dumps([{"image_id":r["image_id"],"split":r["split"],"predictions":{k:v for k,v in r["predictions"].items() if v["label"]=="UNKNOWN"}} for r in records if any(v["label"]=="UNKNOWN" for v in r["predictions"].values())],indent=2)+"\n",encoding="utf-8")
    return summary
