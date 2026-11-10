"""CLIP text embeddings and train-only caption knowledge-base index."""
from __future__ import annotations

import json
import time
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import transformers
from transformers import CLIPModel, CLIPProcessor

def _revision(model_name):
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_name,local_files_only=True)).name
    except Exception: return None


def extract_train(root: Path, config: dict):
    captions=json.loads((root/"data"/"captions"/"train.json").read_text(encoding="utf-8"))
    if len(captions)!=50000: raise RuntimeError(f"Expected exactly 50,000 training captions, found {len(captions)}")
    model_name=config["embedding_models"]["text"]
    revision=_revision(model_name)
    out=root/config.get("text_embedding_output_dir","embeddings/text_embeddings"); out.mkdir(parents=True,exist_ok=True)
    emb_path=out/"train.npy"; done_path=out/"train.done.npy"; meta_path=out/"train.metadata.json"
    try:
        meta=json.loads(meta_path.read_text(encoding="utf-8")); a=np.load(emb_path,mmap_mode="r")
        if (a.shape==(50000,512) and np.isfinite(a).all() and meta.get("model")==model_name
            and meta.get("count")==50000 and meta.get("normalization")=="L2" and meta.get("model_revision")==revision
            and [x["caption_id"] for x in meta.get("captions",[])]==[x["caption_id"] for x in captions]):
            return {"status":"cached","count":50000,"dimension":512,"model":model_name}
    except (OSError,ValueError,json.JSONDecodeError): pass
    device_name=config.get("embedding_device","auto")
    device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else "cpu" if device_name=="auto" else device_name)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but not available")
    try:
        processor=CLIPProcessor.from_pretrained(model_name,local_files_only=True)
        model=CLIPModel.from_pretrained(model_name,local_files_only=True).eval().to(device)
    except Exception as exc: raise RuntimeError(f"Cannot load local CLIP text model '{model_name}': {exc}") from exc
    dim=int(model.config.projection_dim); bs=int(config.get("embedding_batch_size",64))
    if emb_path.exists():
        try:
            matrix=np.load(emb_path,mmap_mode="r+"); done=np.load(done_path) if done_path.exists() else np.zeros(50000,bool)
            if matrix.shape!=(50000,dim) or done.shape!=(50000,): raise ValueError("shape mismatch")
        except (OSError,ValueError):
            emb_path.unlink(missing_ok=True); done_path.unlink(missing_ok=True)
            matrix=np.lib.format.open_memmap(emb_path,mode="w+",dtype=np.float32,shape=(50000,dim)); done=np.zeros(50000,bool)
    else:
        matrix=np.lib.format.open_memmap(emb_path,mode="w+",dtype=np.float32,shape=(50000,dim)); done=np.zeros(50000,bool)
    start_time=time.time(); pending=np.flatnonzero(~done)
    for start in range(0,len(pending),bs):
        ix=pending[start:start+bs]; texts=[captions[int(i)]["caption_raw"] for i in ix]
        inputs=processor(text=texts,return_tensors="pt",padding=True,truncation=True,max_length=77).to(device)
        with torch.inference_mode():
            txt=model.text_model(input_ids=inputs["input_ids"],attention_mask=inputs.get("attention_mask"))
            vec=torch.nn.functional.normalize(model.text_projection(txt.pooler_output).float(),p=2,dim=-1)
        values=vec.cpu().numpy().astype(np.float32,copy=False)
        if values.shape!=(len(ix),dim): raise ValueError(f"Unexpected CLIP text feature shape {values.shape}")
        matrix[ix]=values; matrix.flush(); done[ix]=True; np.save(done_path,done)
        if (start//bs)%20==0: print(f"CLIP text train: {min(start+len(ix),len(pending))}/{len(pending)} pending",flush=True)
    if not done.all(): raise RuntimeError(f"Incomplete text embeddings: {int(done.sum())}/50000")
    mapping=[{"embedding_index":i,"caption_id":int(c["caption_id"]),"image_id":int(c["image_id"]),
              "caption_raw":c["caption_raw"],"caption_cleaned":c["caption_cleaned"]} for i,c in enumerate(captions)]
    meta_obj={"model":model_name,"model_revision":revision,"count":50000,"dimension":dim,"dtype":"float32","normalization":"L2",
              "device":str(device),"batch_size":bs,"torch_version":str(torch.__version__),"transformers_version":transformers.__version__,"max_token_length":77,"text_source":"caption_raw, training split only",
              "elapsed_seconds":round(time.time()-start_time,2),"captions":mapping}
    tmp=meta_path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(meta_obj,ensure_ascii=False)+"\n",encoding="utf-8"); tmp.replace(meta_path)
    (out/"extraction_statistics.json").write_text(json.dumps({k:v for k,v in meta_obj.items() if k!="captions"},indent=2)+"\n",encoding="utf-8")
    return {k:v for k,v in meta_obj.items() if k!="captions"}


def build_caption_index(root: Path, config: dict):
    started=time.time()
    out=root/config.get("text_embedding_output_dir","embeddings/text_embeddings")
    ep=out/"train.npy"; mp=out/"train.metadata.json"
    if not ep.exists() or not mp.exists(): raise RuntimeError("Run `python main.py text_embeddings` before caption_index")
    meta=json.loads(mp.read_text(encoding="utf-8")); matrix=np.load(ep,mmap_mode="r")
    if matrix.shape!=(50000,512) or meta.get("count")!=50000: raise ValueError("Training caption embedding count/dimension mismatch")
    norms=np.linalg.norm(matrix,axis=1)
    if not np.isfinite(matrix).all() or not np.allclose(norms,1,atol=2e-3): raise ValueError("Caption embeddings are invalid or not L2-normalized")
    caps=meta["captions"]
    # Assert every vector maps to the source's train caption order and no heldout IDs.
    source=json.loads((root/"data"/"captions"/"train.json").read_text(encoding="utf-8"))
    if [int(x["caption_id"]) for x in caps] != [int(x["caption_id"]) for x in source]: raise ValueError("Caption vector mapping order differs from frozen training caption data")
    heldout_ids=set()
    for split in ("validation","test"):
        heldout_ids.update(int(x["caption_id"]) for x in json.loads((root/"data"/"captions"/f"{split}.json").read_text(encoding="utf-8")))
    cap_ids={int(x["caption_id"]) for x in caps}
    if cap_ids & heldout_ids: raise RuntimeError("Validation/test caption IDs found in the training KB")
    if len(cap_ids)!=50000: raise RuntimeError(f"Expected 50,000 unique training caption IDs, found {len(cap_ids)}")
    index=faiss.IndexFlatIP(512); index.add(np.asarray(matrix,dtype=np.float32))
    dest=root/"vector_database"; dest.mkdir(parents=True,exist_ok=True)
    idx_path=dest/"caption_text_cosine_train.index"; faiss.write_index(index,str(idx_path))
    description={"technology":"FAISS","index_type":"IndexFlatIP","metric":"cosine similarity via normalized vectors / inner product",
        "faiss_version":faiss.__version__,
        "dimension":512,"ntotal":int(index.ntotal),"source_split":"train","model":meta["model"],
        "model_revision":meta.get("model_revision"),"metadata_file":"../embeddings/text_embeddings/train.metadata.json","contains_validation_or_test_captions":False,
        "caption_id_count":len(cap_ids),"unique_caption_ids":len(cap_ids),"image_id_count":len({int(x["image_id"]) for x in caps})}
    (dest/"caption_text_cosine_train.metadata.json").write_text(json.dumps(description,indent=2)+"\n",encoding="utf-8")
    reread=faiss.read_index(str(idx_path))
    # Retrieval smoke test by one training text vector, excluding that exact vector.
    q=np.asarray(matrix[0:1],dtype=np.float32); scores,positions=reread.search(q,2)
    keep=[int(pos) for pos in positions[0] if int(pos)!=0]
    if reread.ntotal!=50000 or reread.d!=512 or not keep: raise RuntimeError("Caption FAISS read-back/retrieval check failed")
    description["indexing_seconds"]=round(time.time()-started,3); description["index_size_bytes"]=idx_path.stat().st_size
    (dest/"caption_text_cosine_train.metadata.json").write_text(json.dumps(description,indent=2)+"\n",encoding="utf-8")
    result={"status":"PASS","index_type":"IndexFlatIP","vectors":reread.ntotal,"dimension":reread.d,
        "caption_ids_unique":len(cap_ids)==50000,"heldout_caption_ids_in_kb":len(cap_ids&heldout_ids),
        "smoke_query_caption_id":int(caps[0]["caption_id"]),"top_neighbor_caption_id":int(caps[keep[0]]["caption_id"]),
        "top_neighbor_score":float(scores[0][list(positions[0]).index(keep[0])]),"index_path":idx_path.relative_to(root).as_posix()}
    report=root/"reports"/"knowledge_base"; report.mkdir(parents=True,exist_ok=True)
    (report/"caption_index_statistics.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result
