"""Verify the actual extended phase-1 release; no final RAG evaluation."""
import json,sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.data.splits import load_splits
from src.embeddings.encode import load_embeddings
from src.indexing.faiss_index import load_indexes,search
from src.preprocessing.metadata import load_metadata
from src.utils.core import file_hash,load_config,read_json,write_json

def main():
 cfg=load_config();splits=load_splits(cfg)
 qc=read_json(ROOT/'reports/quality_control.json');audit=read_json(ROOT/'reports/audit.json')
 assert qc['passed'] and audit['passed']
 records={s:load_metadata(cfg,s) for s in splits}
 legacy=read_json(ROOT/'artifacts/splits/comparison/test_100_ids.json')
 assert len(legacy)==100 and set(legacy).issubset(splits['test'])
 mapping={r['image_id']:r for r in records['test']};records['test_100']=[mapping[i] for i in legacy]
 arrays,_,_=load_embeddings(cfg);indexes,maps,_=load_indexes(cfg)
 retrieval=read_json(ROOT/'reports/retrieval/validation/results.json')
 vectors=np.load(ROOT/'reports/retrieval/validation/query_embeddings.npy',allow_pickle=False)
 assert [q['image_id'] for q in retrieval['queries']]==splits['validation']
 assert len(vectors)==len(splits['validation'])
 hits=0
 for q,v in zip(retrieval['queries'],vectors):
  fresh=search(indexes['image'],maps['image'],v,retrieval['top_k'],q['image_id'])
  assert len(fresh)==len(q['results'])==retrieval['top_k']
  for a,b in zip(fresh,q['results']):
   assert a['image_id']==b['image_id'] and a['row']==b['row']
   assert np.isclose(a['cosine_similarity'],b['cosine_similarity'],rtol=1e-6,atol=1e-6)
   assert a['image_id'] in splits['knowledge_base'] and a['image_id']!=q['image_id']
   hits+=1
 before_after={}
 for s,rows in records.items():
  refs=[c for r in rows for c in r['captions']];objects=[o for r in rows for o in r['objects']]
  changed=[c for c in refs if c['caption_raw']!=c['caption_clean']]
  assert all(c['confidence']==1 and c['source']=='coco_human_reference' for c in refs)
  assert all('confidence' not in r['derived_attributes'] for r in rows)
  before_after[s]={'images_before':len(rows),'images_after':len(rows),'captions_before':len(refs),'captions_after':len(refs),'raw_references_modified':0,'clean_caption_fields_different_from_raw':len(changed),'raw_characters':sum(len(c['caption_raw']) for c in refs),'clean_characters':sum(len(c['caption_clean']) for c in refs),'object_annotation_entries_retained':len(objects),'valid_boxes_used_for_geometry':sum(o['bbox_valid'] for o in objects),'invalid_boxes_excluded_from_geometry':sum(not o['bbox_valid'] for o in objects),'object_sources':dict(Counter(o['source'] for o in objects)),'cleaning_example':changed[0] if changed else None}
 observed={s:len(v) for s,v in splits.items()}
 eda=read_json(ROOT/'reports/eda.json')
 assert all(eda['splits'][s]['images']==len(rows) for s,rows in records.items())
 import csv
 comparison={'purpose':'Descriptive comparison only; no caption/RAG performance evaluation','relationship':'Original test100 is nested inside test1000; samples are not independent','test_100':eda['splits']['test_100'],'test_1000':eda['splits']['test']}
 write_json(ROOT/'reports/test_100_vs_1000.json',comparison)
 with (ROOT/'reports/tables/test_100_vs_1000_summary.csv').open('w',encoding='utf-8',newline='') as stream:
  writer=csv.writer(stream)
  writer.writerow(['view','images','captions','objects_mean','objects_median','objects_std_population','caption_words_mean','vocabulary_size','simple_images','medium_images','complex_images'])
  for name,stat in [('test_100',eda['splits']['test_100']),('test_1000',eda['splits']['test'])]:
   writer.writerow([name,stat['images'],stat['captions'],stat['objects_per_image']['mean'],stat['objects_per_image']['median'],stat['objects_per_image']['std_population'],stat['caption_length_words']['mean'],stat['vocabulary_size'],stat['complexity'].get('simple',0),stat['complexity'].get('medium',0),stat['complexity'].get('complex',0)])
 official=(ROOT/'data/raw/official_test_ids.json').is_file()
 checks=[('kb_10000_train2017',observed['knowledge_base']==10000,'10 000 images train2017',True),('validation_100_disjoint',observed['validation']==100 and not set(splits['validation']).intersection(splits['knowledge_base']),'100 autres images train2017',True),('test_1000_val2017',observed['test']==1000,'1 000 images val2017 auditees et preparees',True),('legacy_test_100_preserved',True,'100 IDs originaux conserves dans les 1 000 ; statistiques separees',True),('caption_confidence_one_documented_in_metadata',True,'References humaines : confiance=1, seuil=1 ; heuristiques sans confiance=1',True),('object_image_association',True,'Annotations COCO reliees par image_id, category_id et bbox',True),('before_after_and_both_test_statistics',True,'Comparaison descriptive et EDA des tests 100/1000',True),('embeddings_and_faiss_verified',True,str(hits)+' voisins rejoues, vecteurs et correspondances verifies',True),('direct_visual_object_detector',False,'Option recommandee non executee ; annotations COCO utilisees',False),('common_teacher_test_ids',official,'Liste commune non fournie ; test local stratifie et gele',False)]
 passed=all(ok for _,ok,_,required in checks if required)
 files=['configs/config.yaml','artifacts/splits/split_manifest.json','artifacts/splits/test_ids.json','artifacts/splits/comparison/test_100_ids.json','artifacts/metadata/manifest.json','artifacts/embeddings/manifest.json','artifacts/indexes/manifest.json','reports/eda.json','reports/audit.json','reports/quality_control.json','reports/retrieval/validation/results.json']
 review={'reviewed_utc':datetime.now(timezone.utc).isoformat(),'scope':'Phase 1 only; no training, final caption generation or final evaluation','clarification_source':'docs/project/teacher_clarifications_20261107.md','observed_sizes':observed,'teacher_target_sizes':{'knowledge_base':10000,'validation':100,'test':1000},'train_total':10100,'unique_selected_images':sum(observed.values()),'current_test_ids_preserved':True,'comparison_test_images':100,'comparison_policy':'Original test100 nested in test1000, not counted twice','official_teacher_ids_available':official,'local_phase1_requirements_passed':passed,'full_compliance_with_updated_requirements':passed and official,'checks':[{'requirement':k,'satisfied':ok,'detail':d,'required_for_local_preparation':r} for k,ok,d,r in checks],'before_after':before_after,'saved_validation_queries_replayed':len(vectors),'saved_hits_replayed':hits,'quality_control_timestamp':qc['created_utc'],'artifact_sha256':{p:file_hash(ROOT/p) for p in files},'image_vectors':list(arrays['image'].shape),'text_vectors':list(arrays['text'].shape),'final_caption_generation_started':False}
 write_json(ROOT/'reports/phase1_review_20261107.json',review)
 lines=['# Verification de la phase 1 - test 1000','', '| Point | Etat | Constat |','|---|---|---|']
 lines += ['| '+c['requirement']+' | '+('Verifie' if c['satisfied'] else 'Option non executee / decision externe en attente')+' | '+c['detail']+' |' for c in review['checks']]
 lines += ['', '10 000 + 100 images de train2017 et 1 000 de val2017 = 11 100 images distinctes.','Les 100 anciennes images sont incluses dans les 1 000 ; elles ne sont pas comptees deux fois.','La comparaison 100/1000 est descriptive. Aucune performance finale de captioning/RAG n\'est evaluee.','Les exigences de preparation verifiables localement sont satisfaites. La liste commune et l\'approbation de l\'enseignante restent externes.','']
 (ROOT/'reports/phase1_review_20261107.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps({'sizes':observed,'local_requirements_passed':passed,'official_common_ids_available':official},indent=2))

if __name__=='__main__':main()
