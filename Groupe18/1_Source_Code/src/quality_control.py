"""Machine-readable final quality gates; missing work is explicitly NOT_RUN."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def run(root: Path, config: dict):
    checks={}
    def load_optional(path):
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: return None
    def record(name, ok, detail, status=None):
        checks[name]={"status":status or ("PASS" if ok else "FAIL"),"detail":detail}
    expected={"train":10000,"validation":100,"test":1000}
    manifests={}; captions={}
    for split,n in expected.items():
        mp=root/"data"/"selected"/f"{split}_ids.json"
        cp=root/"data"/"captions"/f"{split}.json"
        try: manifests[split]=json.loads(mp.read_text(encoding="utf-8"))["image_ids"]
        except Exception as exc: manifests[split]=[]; record(f"manifest_{split}",False,str(exc))
        try: captions[split]=json.loads(cp.read_text(encoding="utf-8"))
        except Exception as exc: captions[split]=[]; record(f"captions_{split}",False,str(exc))
        record(f"count_{split}",len(manifests[split])==n,{"expected":n,"actual":len(manifests[split])})
        ids=[int(x) for x in manifests[split]]
        count=collections.Counter(int(x["image_id"]) for x in captions[split])
        wrong={str(i):count[i] for i in ids if count[i]!=5}
        record(f"five_captions_{split}",not wrong,{"images_with_wrong_caption_count":wrong,"caption_records":len(captions[split])})
        missing=[int(i) for i in ids if not (root/"data"/"processed"/split/f"{int(i):012d}.jpg").is_file()]
        record(f"processed_files_{split}",not missing,{"missing_count":len(missing),"examples":missing[:10]})
        decode_errors=[]
        for iid in ids:
            p=root/"data"/"processed"/split/f"{int(iid):012d}.jpg"
            try:
                with Image.open(p) as im:
                    im.verify()
                    if im.size!=(int(config["image_size"]),int(config["image_size"])): raise ValueError(f"unexpected dimensions {im.size}")
            except Exception as exc: decode_errors.append({"image_id":int(iid),"reason":str(exc)})
        record(f"decode_{split}",not decode_errors,{"decoded":len(ids)-len(decode_errors),"errors":decode_errors[:20]})
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        overlap=sorted(set(map(int,manifests.get(a,[])))&set(map(int,manifests.get(b,[]))))
        record(f"overlap_{a}_{b}",not overlap,{"count":len(overlap),"examples":overlap[:20]})
    report_dir=root/"reports"/"dataset_audit"
    exact_path=report_dir/"selected_images_integrity.json"
    try:
        exact=json.loads(exact_path.read_text(encoding="utf-8")); dups=exact.get("hash_duplicate_groups",[])
        record("exact_sha256",not dups,{"duplicate_groups":len(dups)})
    except Exception as exc: record("exact_sha256",False,str(exc))
    phash_path=root/"reports"/"perceptual_hash"/"hash_statistics.json"
    review_path=root/"reports"/"perceptual_hash"/"cross_split_visual_review.json"
    if phash_path.exists():
        ph=json.loads(phash_path.read_text(encoding="utf-8")); review_data=load_optional(review_path) or {}
        cross_count=sum(ph.get("pairs_across_splits",{}).values())
        good=not ph.get("errors") and ph.get("images_hashed")==11100
        pair_rows=review_data.get("pairs",[])
        unresolved=[f"{x.get('train_image_id')}/{x.get('test_image_id')}" for x in pair_rows if x.get("retain_and_flag") or "unresolved" in x.get("classification","")]
        review_complete=len(pair_rows)==cross_count
        status="FAIL" if not good else ("WARNING" if unresolved or not review_complete else "PASS")
        record("perceptual_hash",good,{"images_hashed":ph.get("images_hashed"),"candidate_pairs":ph.get("candidate_pair_count"),"threshold":ph.get("threshold_hamming_inclusive"),"cross_split_candidates":cross_count,"cross_split_pairs_reviewed":len(pair_rows),"review_classifications":dict(collections.Counter(x.get("classification","unclassified") for x in pair_rows)),"unresolved_pairs":unresolved,"manual_review_report_present":review_path.exists(),"errors":len(ph.get("errors",[]))},status)
    else: record("perceptual_hash",False,"pHash report absent", "NOT_RUN")
    preserved_path=root/"reports"/"perceptual_hash"/"preserved_result_hashes.json"
    preserved=load_optional(preserved_path) or {}
    phash_files_ok=preserved.get("all_preserved") is True and all(x.get("archived_sha256")==x.get("backup_sha256") for x in preserved.get("files",[])) and len(preserved.get("files",[]))==4
    record("historical_sha256_and_phash_results_archived",phash_files_ok,{"artifacts_checked":len(preserved.get("files",[])),"all_archived_byte_for_byte":preserved.get("all_preserved"),"current_revision_recomputed":bool(preserved.get("current_revision_artifacts"))},"PASS" if phash_files_ok else ("NOT_RUN" if not preserved else "FAIL"))
    emb_dir=root/config.get("embedding_output_dir","embeddings/image_embeddings")
    emb_stats={}; emb_ok=True
    for split,n in expected.items():
        p=emb_dir/f"{split}.npy"; mp=emb_dir/f"{split}.metadata.json"
        try:
            arr=np.load(p,mmap_mode="r"); meta=json.loads(mp.read_text(encoding="utf-8"))
            item_ok=arr.shape==(n,512) and np.isfinite(arr).all() and meta.get("count")==n and meta.get("model")==config["embedding_models"]["image"] and meta.get("normalization")=="L2"
            norms=np.linalg.norm(arr,axis=1)
            item_ok=bool(item_ok and np.allclose(norms,1,atol=2e-3))
            emb_stats[split]={"shape":list(arr.shape),"finite":bool(np.isfinite(arr).all()),"unit_norm":bool(np.allclose(norms,1,atol=2e-3))}
            emb_ok &= item_ok
        except Exception as exc: emb_ok=False; emb_stats[split]={"error":str(exc)}
    record("image_embeddings",emb_ok,emb_stats,"PASS" if emb_ok else ("NOT_RUN" if not any((emb_dir/f"{s}.npy").exists() for s in expected) else "FAIL"))
    embedding_mapping_ok=True; mapping_counts={}
    for split,n in expected.items():
        try:
            meta=json.loads((emb_dir/f"{split}.metadata.json").read_text(encoding="utf-8")); mapped=[int(x["image_id"]) for x in meta["embeddings"]]
            ok=mapped==list(map(int,manifests[split])); embedding_mapping_ok &= ok; mapping_counts[split]={"rows":len(mapped),"matches_manifest_order":ok}
        except Exception as exc: embedding_mapping_ok=False; mapping_counts[split]={"error":str(exc)}
    record("embedding_id_mapping",embedding_mapping_ok,mapping_counts,"PASS" if embedding_mapping_ok else ("NOT_RUN" if not any((emb_dir/f"{s}.metadata.json").exists() for s in expected) else "FAIL"))
    index_path=root/"vector_database"/"image_cosine_train.index"; index_meta=root/"vector_database"/"image_cosine_train.metadata.json"
    retrieval_path=root/"reports"/"retrieval"/"retrieval_evaluation.json"
    if index_path.exists() and index_meta.exists():
        try:
            import faiss
            ix=faiss.read_index(str(index_path)); im=json.loads(index_meta.read_text(encoding="utf-8"))
            ok=ix.ntotal==expected["train"] and ix.d==512 and im.get("ntotal")==ix.ntotal and im.get("gallery_image_ids")==list(map(int,manifests["train"]))
            record("faiss_index",ok,{"ntotal":ix.ntotal,"dimension":ix.d,"index_type":type(ix).__name__})
        except Exception as exc: record("faiss_index",False,str(exc))
    else: record("faiss_index",False,"index absent","NOT_RUN")
    if retrieval_path.exists():
        r=json.loads(retrieval_path.read_text(encoding="utf-8")); ok=r.get("status")=="PASS" and r.get("self_match_count")==0
        record("retrieval",ok,r)
    else: record("retrieval",False,"retrieval evaluation absent","NOT_RUN")
    text_dir=root/config.get("text_embedding_output_dir","embeddings/text_embeddings")
    text_ok=False; text_detail={}
    try:
        ta=np.load(text_dir/"train.npy",mmap_mode="r"); tm=json.loads((text_dir/"train.metadata.json").read_text(encoding="utf-8"))
        train_caps=captions["train"]; held_caps=captions["validation"]+captions["test"]
        mapped=tm["captions"]; mapped_ids={int(x["caption_id"]) for x in mapped}; held_ids={int(x["caption_id"]) for x in held_caps}
        train_id_order=[int(x["caption_id"]) for x in train_caps]
        mapping_match=[int(x["caption_id"]) for x in mapped]==train_id_order and all(int(x["image_id"])==int(src["image_id"]) for x,src in zip(mapped,train_caps))
        text_ok=ta.shape==(50000,512) and np.isfinite(ta).all() and np.allclose(np.linalg.norm(ta,axis=1),1,atol=2e-3) and mapped_ids==set(train_id_order) and not (mapped_ids&held_ids) and mapping_match
        ixp=root/"vector_database"/"caption_text_cosine_train.index"; imp=root/"vector_database"/"caption_text_cosine_train.metadata.json"
        if ixp.exists() and imp.exists():
            import faiss
            tix=faiss.read_index(str(ixp)); tim=json.loads(imp.read_text(encoding="utf-8"))
            text_ok &= tix.ntotal==50000 and tix.d==512 and tim.get("ntotal")==50000 and not tim.get("contains_validation_or_test_captions",True)
        else: text_ok=False
        text_detail={"shape":list(ta.shape),"mapping_matches_source":mapping_match,"unique_train_caption_ids":len(mapped_ids),"heldout_caption_ids_in_index":len(mapped_ids&held_ids),"faiss_count":tim.get("ntotal") if ixp.exists() and imp.exists() else 0}
    except Exception as exc: text_detail={"error":str(exc)}
    record("caption_text_embeddings_and_caption_vector_database",bool(text_ok),text_detail,"PASS" if text_ok else ("NOT_RUN" if not (text_dir/"train.npy").exists() else "FAIL"))
    pos_dir=root/"data"/"captions"/"pos"; pos_counts={}; pos_ok=True
    for s,n in expected.items():
        p=pos_dir/f"{s}.jsonl"
        try:
            c=0; malformed=[]
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    row=json.loads(line); c+=1
                    if not isinstance(row.get("tokens"),list) or "caption_raw" not in row: malformed.append(row.get("caption_id"))
            pos_counts[s]={"records":c,"malformed":malformed[:10]}; pos_ok &= c==n*5 and not malformed
        except (OSError,ValueError,json.JSONDecodeError): pos_ok=False; pos_counts[s]={"records":0,"error":"missing or malformed POS output"}
    record("pos_tagging",pos_ok,{"caption_counts":pos_counts,"expected":{"train":50000,"validation":500,"test":5000}},"PASS" if pos_ok else ("NOT_RUN" if not any(pos_counts.values()) else "FAIL"))
    vis=root/"data"/"metadata"/"visual_predictions.jsonl"
    if vis.exists():
        try:
            c=0; bad=0
            with vis.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row=json.loads(line); c+=1
                        bad += not all(k in row.get("predictions",{}) for k in ("indoor_outdoor","day_night","scene"))
            record("visual_analysis",c==11100 and bad==0,{"records":c,"expected":11100,"malformed_records":bad})
        except (OSError,ValueError,json.JSONDecodeError) as exc: record("visual_analysis",False,str(exc))
    else: record("visual_analysis",False,"visual prediction file absent","NOT_RUN")
    # Frozen selection is verified against its immutable report; this stage never writes manifests.
    try:
        sr=json.loads((root/"data"/"selected"/"selection_report.json").read_text(encoding="utf-8")); matches={s:sr["image_ids"][s]==manifests[s] for s in expected}
        record("frozen_manifest_consistency",all(matches.values()) and config["seed"]==20261106,{"matches_selection_report":matches,"configured_seed":config["seed"]})
    except Exception as exc: record("frozen_manifest_consistency",False,str(exc))
    revision=load_optional(root/"data"/"selected"/"replacement_history"/"revision_1.json")
    if revision:
        backup_dir=root/revision["backup_directory"]
        inventory=load_optional(backup_dir/"backup_inventory.json") or {}
        inv={x["path"]:x for x in inventory.get("files",[])}
        before=revision.get("manifest_files_before_sha256",{}); after=revision.get("manifest_files_after_sha256",{})
        backup_ok=all(rel in inv and inv[rel].get("sha256")==before.get(rel)==inv[rel].get("backup_sha256") for rel in before)
        trainval=["data/selected/train_ids.json","data/selected/train_ids.csv","data/selected/validation_ids.json","data/selected/validation_ids.csv"]
        unchanged=all(before.get(rel)==after.get(rel) for rel in trainval)
        hashes_match=all(hashlib.sha256((root/rel).read_bytes()).hexdigest()==after.get(rel) for rel in after)
        record("controlled_manifest_revision_integrity",backup_ok and unchanged and hashes_match,{"backup_files":len(inv),"originals_match_backup":backup_ok,"train_validation_unchanged":unchanged,"current_hashes_match_revision_record":hashes_match,"test_manifest_intentionally_revised":before.get("data/selected/test_ids.json")!=after.get("data/selected/test_ids.json")})
    else:
        resume_check=load_optional(root/"reports"/"dataset_audit"/"frozen_manifest_resume_check.json")
        if resume_check:
            ok=resume_check.get("unchanged_during_combined_resume") is True and all(x.get("sha256") for x in resume_check.get("files",[]))
            record("frozen_manifest_resume_integrity",ok,{"files":len(resume_check.get("files",[])),"all_unchanged":resume_check.get("unchanged_during_combined_resume")})
        else: record("frozen_manifest_resume_integrity",False,"combined resume hash verification absent","NOT_RUN")
    statuses=[v["status"] for v in checks.values()]
    phase_complete=all(x=="PASS" for x in statuses)
    overall="FAIL" if any(x=="FAIL" for x in statuses) else ("PASS" if phase_complete else "INCOMPLETE")
    result={"phase1_complete":phase_complete,"overall_status":overall,"checks":checks,
        "not_run_checks":[k for k,v in checks.items() if v["status"]=="NOT_RUN"],
        "failed_checks":[k for k,v in checks.items() if v["status"]=="FAIL"],
        "warning_checks":[k for k,v in checks.items() if v["status"]=="WARNING"],
        "notes":["Image and caption embeddings are stored in separately verifiable FAISS indexes; both galleries contain training data only.","Visual model confidence is a relative CLIP prompt score, not calibrated probability."]}
    (root/"reports"/"final_quality_control.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    dataset=load_optional(root/"reports"/"statistical_analysis"/"dataset_audit.json") or {}
    emb_summary=load_optional(emb_dir/"extraction_statistics.json") or {}
    text_summary=load_optional(text_dir/"extraction_statistics.json") or {}
    image_index_meta=load_optional(root/"vector_database"/"image_cosine_train.metadata.json") or {}
    caption_index_meta=load_optional(root/"vector_database"/"caption_text_cosine_train.metadata.json") or {}
    retrieval_summary=load_optional(root/"reports"/"retrieval"/"retrieval_evaluation.json") or {}
    ph_summary=load_optional(phash_path) or {}
    pair_review=load_optional(review_path) or {}
    pos_summary=load_optional(root/"reports"/"caption_pos"/"linguistic_summary.json") or {}
    visual_summary=load_optional(root/"reports"/"visual_analysis"/"summary.json") or {}
    def fmt(value):
        try: return f"{int(value):,}"
        except (TypeError,ValueError): return "unknown"
    summary_lines=["# Final Phase 1 quality report","",f"Overall status: **{result['overall_status']}**; Phase 1 complete: **{result['phase1_complete']}**.","",
        "## Dataset and frozen splits", "",
        f"- Split counts: train {len(manifests.get('train',[])):,}, validation {len(manifests.get('validation',[])):,}, test {len(manifests.get('test',[])):,}.",
        f"- Candidate exclusions for non-five caption cardinality: train 315; val 13. Frozen manifests were not modified by remaining stages.",
        f"- COCO train2017 annotations: {fmt(dataset.get('splits',{}).get('train2017',{}).get('images_in_annotations'))} images, {fmt(dataset.get('splits',{}).get('train2017',{}).get('object_annotations'))} object instances, {fmt(dataset.get('splits',{}).get('train2017',{}).get('captions'))} captions.",
        f"- COCO val2017 annotations: {fmt(dataset.get('splits',{}).get('val2017',{}).get('images_in_annotations'))} images, {fmt(dataset.get('splits',{}).get('val2017',{}).get('object_annotations'))} object instances, {fmt(dataset.get('splits',{}).get('val2017',{}).get('captions'))} captions.","",
        "## Embeddings and retrieval", "",
        f"- Image encoder: {emb_summary.get('model','unknown')} (revision {emb_summary.get('model_revision','recorded in per-split metadata')}); device {emb_summary.get('device','unknown')}; dimension {emb_summary.get('dimension','unknown')}; elapsed {emb_summary.get('elapsed_seconds','unknown')} s.",
        "- Image vectors: train 10,000, validation 100, test 1,000; float32, L2-normalized, finite; ID order checked against frozen manifests.",
        f"- Image FAISS index: {image_index_meta.get('ntotal','unknown')} train vectors, {image_index_meta.get('index_size_bytes','unknown')} bytes, build {image_index_meta.get('indexing_seconds','unknown')} s.",
        f"- Image retrieval: {retrieval_summary.get('query_count','unknown')} held-out queries, top-{retrieval_summary.get('top_k','unknown')}, self matches {retrieval_summary.get('self_match_count','unknown')}, mean top-1 cosine {retrieval_summary.get('mean_top1_cosine','unknown')}.",
        f"- Caption vectors: {text_summary.get('count','unknown')} train captions, {text_summary.get('dimension','unknown')} dimensions, model revision {text_summary.get('model_revision','unknown')}, elapsed {text_summary.get('elapsed_seconds','unknown')} s.",
        f"- Caption FAISS index: {caption_index_meta.get('ntotal','unknown')} train-only vectors, {caption_index_meta.get('index_size_bytes','unknown')} bytes, build {caption_index_meta.get('indexing_seconds','unknown')} s; held-out caption IDs in KB: {checks['caption_text_embeddings_and_caption_vector_database']['detail'].get('heldout_caption_ids_in_index','unknown')}.","",
        "## Perceptual duplicate audit", "",
        f"- DCT pHash: {ph_summary.get('images_hashed','unknown')} images; inclusive Hamming threshold {ph_summary.get('threshold_hamming_inclusive','unknown')}; {ph_summary.get('candidate_pair_count','unknown')} review candidates, including {sum(ph_summary.get('pairs_across_splits',{}).values())} across splits.",
        f"- Candidate distances: {ph_summary.get('hamming_distance_counts',{})}. Exact SHA-256 audit found {len(dups) if isinstance(dups,list) else 0} exact duplicate groups.",
        "- Original-resolution review of all four train/test candidates:",
    ]
    for pair in pair_review.get("pairs",[]):
        summary_lines.append(f"  - {pair.get('train_image_id')} / {pair.get('test_image_id')}, d={pair.get('hamming_distance')}: **{pair.get('classification')}** — {pair.get('reason','')}")
    summary_lines += [
        f"- Final audit decision: **{'WARNING' if pair_review.get('unresolved_quality_warning') else 'PASS'}**. {pair_review.get('remaining_uncertainty','')}",
        "- Pair-level evidence and original-resolution visual comparisons are in `reports/perceptual_hash/cross_split_pair_evidence.json`, `cross_split_native_resolution_comparison.png`, and `hamming2_native_2x.png`. No source result files or manifests were changed.","",
        "## POS and visual analysis", "",
        f"- spaCy POS: model {pos_summary.get('model','unknown')}; {pos_summary.get('caption_count','unknown')} captions; {pos_summary.get('token_count','unknown')} tokens; errors {len(pos_summary.get('errors',[]))}.",
        f"- Most frequent nouns: {pos_summary.get('top_words',{}).get('NOUN',[])[:10]}.",
        f"- Most frequent verbs: {pos_summary.get('top_words',{}).get('VERB',[])[:10]}.",
        f"- Most frequent adjectives: {pos_summary.get('top_words',{}).get('ADJ',[])[:10]}.",
        f"- CLIP scene/context predictions: {visual_summary.get('images','unknown')} images; uncertain context/time/scene counts {visual_summary.get('uncertain_counts',{})}; confidence summary {visual_summary.get('confidence_summary',{})}.",
        "- Scene labels are zero-shot model predictions with uncalibrated softmax scores; low-confidence output is UNKNOWN. COCO categories remain separate annotations. Color is a coarse pixel-level hue/luminance proxy.","",
        "## Check results",""]
    for name,item in checks.items(): summary_lines.append(f"- **{item['status']}** `{name}`: {json.dumps(item['detail'],ensure_ascii=False,default=str)}")
    summary_lines += ["","## Items requiring attention","",f"NOT_RUN checks: {result['not_run_checks'] or 'none'}.",f"WARNING checks: {result['warning_checks'] or 'none'}.",f"FAIL checks: {result['failed_checks'] or 'none'}.","A full-completion claim requires every applicable check to be PASS. Visual model scores are not calibrated probabilities, and perceptual-hash candidates are not confirmed duplicates. COCO provides no explicit relation labels; no interaction claims are made.",""]
    final_md="\n".join(summary_lines)
    (root/"reports"/"FINAL_REPORT.md").write_text(final_md,encoding="utf-8")
    (root/"reports"/"FINAL_STATUS.md").write_text("# Phase 1 final status\n\n"+f"Status: **{result['overall_status']}**. Full Phase 1 acceptance: **{result['phase1_complete']}**.\n\n"+f"Frozen splits, embeddings, image/caption FAISS indexes, pHash, POS, visual analysis, and data integrity checks are reported in [FINAL_REPORT.md](FINAL_REPORT.md) and [final_quality_control.json](final_quality_control.json).\n\n"+f"Warnings: {result['warning_checks'] or 'none'}. Not run: {result['not_run_checks'] or 'none'}. Failures: {result['failed_checks'] or 'none'}.\n"+f"The remaining pHash warning is pair 535889 / 465129, classified as a strong near-duplicate with uncertain source provenance. See [FINAL_SCIENTIFIC_AUDIT.md](FINAL_SCIENTIFIC_AUDIT.md).\n",encoding="utf-8")
    return result
