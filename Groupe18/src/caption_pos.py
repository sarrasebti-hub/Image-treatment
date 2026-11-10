"""Batch spaCy POS annotation of the frozen reference captions."""
from __future__ import annotations

import collections
import json
from pathlib import Path


def run(root: Path, config: dict):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("POS tagging requires spaCy. Install requirements.txt and en_core_web_sm.") from exc
    model_name=config["pos"].get("spacy_model","en_core_web_sm")
    try:
        nlp=spacy.load(model_name,disable=["ner"])
    except OSError as exc:
        raise RuntimeError(f"spaCy model '{model_name}' unavailable; run `python -m spacy download {model_name}`") from exc
    out=root/"data"/"captions"/"pos"; out.mkdir(parents=True,exist_ok=True)
    bsize=int(config["pos"].get("batch_size",1000))
    files=[]
    for split in ("train","validation","test"):
        source=root/"data"/"captions"/f"{split}.json"
        records=json.loads(source.read_text(encoding="utf-8"))
        dest=out/f"{split}.jsonl"; partial=out/f"{split}.partial.jsonl"; meta=out/f"{split}.metadata.json"
        cached_rows={}
        if dest.exists() and meta.exists():
            try:
                m=json.loads(meta.read_text(encoding="utf-8"))
                if m.get("model")==model_name:
                    with dest.open(encoding="utf-8") as f:
                        for line in f:
                            if not line.strip(): continue
                            row=json.loads(line); cached_rows[int(row["caption_id"])]=row
            except (OSError,ValueError,json.JSONDecodeError): cached_rows={}
        done={}
        if partial.exists():
            with partial.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row=json.loads(line); done[int(row["caption_id"])]=row
                    except (ValueError,KeyError,json.JSONDecodeError): continue
        by_id={int(r["caption_id"]):r for r in records}
        for cid,row in cached_rows.items():
            src=by_id.get(cid)
            if src and int(row.get("image_id",-1))==int(src["image_id"]) and row.get("caption_raw")==src.get("caption_raw"):
                done[cid]=row
        remaining=[r for r in records if int(r["caption_id"]) not in done]
        with partial.open("a",encoding="utf-8") as f:
            for start in range(0,len(remaining),bsize):
                batch=remaining[start:start+bsize]
                valid=[]; contexts=[]
                for row in batch:
                    raw=row.get("caption_raw")
                    if not isinstance(raw,str) or not raw.strip():
                        done[int(row["caption_id"]) ]={**row,"pos_error":"missing or non-text caption","tokens":[]}
                    else:
                        valid.append(raw); contexts.append(row)
                for doc,row in nlp.pipe(zip(valid,contexts),as_tuples=True,batch_size=min(bsize,256)):
                    tokens=[{"text":t.text,"lemma":t.lemma_,"pos":t.pos_,"tag":t.tag_,"is_stop":bool(t.is_stop)} for t in doc if not t.is_space]
                    done[int(row["caption_id"]) ]={**row,"tokens":tokens}
                # checkpoint all newly completed annotations; deterministic by ID
                written=set()
                for row in batch:
                    cid=int(row["caption_id"])
                    if cid in done and cid not in written:
                        f.write(json.dumps(done[cid],ensure_ascii=False)+"\n"); written.add(cid)
                f.flush()
                print(f"POS {split}: {min(start+len(batch),len(remaining))}/{len(remaining)} new captions",flush=True)
        if len(done)!=len(by_id):
            raise RuntimeError(f"POS checkpoint incomplete for {split}: {len(done)}/{len(by_id)}")
        # Keep output in original source order, independent of checkpoint resume order.
        tmp=out/f"{split}.jsonl.tmp"
        with tmp.open("w",encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(done[int(row["caption_id"])],ensure_ascii=False)+"\n")
        tmp.replace(dest); partial.unlink(missing_ok=True)
        meta.write_text(json.dumps({"model":model_name,"count":len(records),"source":source.relative_to(root).as_posix(),"preserves_caption_raw":True},indent=2)+"\n",encoding="utf-8")
        files.append(dest)
    word_counts={x:collections.Counter() for x in ("NOUN","VERB","ADJ","ADV","PROPN")}
    pos_counts=collections.Counter(); token_total=0; errors=[]; caption_total=0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            row=json.loads(line); caption_total+=1
            if row.get("pos_error"): errors.append({"caption_id":row["caption_id"],"reason":row["pos_error"]})
            for t in row.get("tokens",[]):
                token_total+=1; pos_counts[t["pos"]]+=1
                if t["pos"] in word_counts: word_counts[t["pos"]][t["lemma"].casefold()]+=1
    outreport=root/"reports"/"caption_pos"; outreport.mkdir(parents=True,exist_ok=True)
    distribution=[{"pos":k,"tokens":v,"percent":100*v/max(token_total,1)} for k,v in pos_counts.most_common()]
    import pandas as pd
    pd.DataFrame(distribution).to_csv(outreport/"pos_distribution.csv",index=False)
    tops={pos:freq.most_common(50) for pos,freq in word_counts.items()}
    (outreport/"linguistic_summary.json").write_text(json.dumps({"model":model_name,"spacy_version":spacy.__version__,"caption_count":caption_total,"token_count":token_total,"pos_distribution":distribution,"top_words":tops,"errors":errors,"limitations":"spaCy statistical English model output; POS tags are model predictions, not manually verified ground truth."},indent=2)+"\n",encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig,ax=plt.subplots(figsize=(9,5)); sns.barplot(data=pd.DataFrame(distribution).head(15),x="tokens",y="pos",color="#4f83a1",ax=ax); ax.set(title="POS tag distribution across selected COCO captions",xlabel="Tokens",ylabel="spaCy POS"); fig.tight_layout(); fig.savefig(root/"visualizations"/"caption_pos_distribution.png",dpi=180); plt.close(fig)
    except ImportError: pass
    return {"status":"PASS" if not errors and caption_total==55500 else "WARNING","model":model_name,"captions":caption_total,"tokens":token_total,"errors":len(errors),"top_words":tops}
