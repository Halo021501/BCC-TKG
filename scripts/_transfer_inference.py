"""Transfer inference extracted from the recorded runner (no fitting).

Portability change: caller supplies CODE, OUT, device, K and persistence helpers;
the implicit Ollama unload request has been removed. Scientific scoring is unchanged.
"""
from pathlib import Path
import bisect, json, sqlite3, time
from collections import defaultdict
import torch
from cbrc.adapter import load_frozen
from cbrc.core import CandidateCorrector, PublicQuery, CausalCandidateHistory, candidate_features, corrected_scores, average_rank
from cbrc.pipeline import rank_metrics
import src.data as data
import src.train as train
from src.llm_cache import LLMEvidenceCache
def head(asset,variant):
 x=asset['heads'][variant];saved=torch.load(x['path'],map_location='cpu',weights_only=True)
 assert saved['variant']==variant and saved['scale']==x['scale']
 m=CandidateCorrector(variant);m.load_state_dict(saved['model'],strict=True);m.eval().requires_grad_(False)
 assert sum(p.numel() for p in m.parameters())==193
 return m,float(x['scale'])

def evaluate(kg,protocol,selected,p,asset):
 seed=asset['seed'];dest=OUT/f'seed{seed}';dest.mkdir(exist_ok=True);end=dest/'evaluation.json'
 if end.exists():assert read(end)['protocol_sha256']==sha(OUT/'protocol.json');return read(end)
 torch.set_num_threads(1);train.set_seed(seed);device=torch.device(DEVICE);model,cfg=load_frozen(CODE,asset['graph_checkpoint'],kg,device)
 cm,cs=head(asset,'cbrc');gm,gs=head(asset,'graph_control')
 evidence=LLMEvidenceCache(str(p),max_candidates=10,expected_shot=K,expected_split='test',expected_history_protocol='standard_rolling_history',expected_dataset_fingerprint=protocol['dataset_fingerprint'],require_generation_metadata=True)
 initial=kg.train_aug+kg.valid_aug;history=data.HistoryIndex(initial,kg.num_entities,kg.num_relations*2,cfg['history_len'],adaptive_decay=not cfg.get('disable_temporal_calibration',False));support_by_rel=train.group_by_relation(initial)
 ch=CausalCandidateHistory(kg.train_aug)
 for s,r,v,t in kg.valid_aug:bisect.insort(ch.times[s,r,v],t)
 filters=data.build_filter_dict(data.add_inverse(kg.train+kg.valid+kg.test,kg.num_relations))
 snapshots=defaultdict(list)
 for fact in kg.test:snapshots[fact[3]].append(fact)
 oriented_selected=set(data.add_inverse(sorted(selected),kg.num_relations))
 db=sqlite3.connect(dest/'ranks.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE IF NOT EXISTS config(signature TEXT PRIMARY KEY)');db.execute('CREATE TABLE IF NOT EXISTS ranks(key TEXT PRIMARY KEY,payload TEXT)')
 signature=sha(OUT/'protocol.json');saved=db.execute('SELECT signature FROM config').fetchall()
 if saved:assert saved==[(signature,)]
 else:db.execute('INSERT INTO config VALUES (?)',(signature,));db.commit()
 done={r[0] for r in db.execute('SELECT key FROM ranks')};started=time.time()
 with torch.inference_mode():
  for t in sorted(snapshots):
   ch.begin_timestamp(t);all_rows=data.add_inverse(snapshots[t],kg.num_relations)
   rows=[f for f in all_rows if tuple(f) in oriented_selected];groups=train.group_by_relation(rows)
   for relation in sorted(groups):
    relrows=groups[relation]
    for offset in range(0,len(relrows),64):
     batch=[f for f in relrows[offset:offset+64] if ':'.join(map(str,f)) not in done]
     if not batch:continue
     public=[PublicQuery(s,r,t) for s,r,o,t in batch];sanitized=[(q.known_entity_id,q.oriented_relation_id,0,q.timestamp) for q in public]
     sup=train.causal_support_rows(support_by_rel,sanitized,K);assert all(f[3]<t for ss in sup for f in ss)
     feat=train.move_features(evidence.augment_features(sanitized,history.build(sanitized)),device)
     assert int(feat['llm_cache_hit'].sum())==len(batch)
     qt=data.tensorize(sanitized,device);st=train.support_rows_tensor(sup,device)
     model.set_llm_runtime_mode('off');base,aux=model(qt,st,feat)
     model.set_llm_runtime_mode('rationale');legacy,_=model(qt,st,feat);model.set_llm_runtime_mode('off')
     ids=feat['llm_candidate_ids'];mask=feat['llm_candidate_mask'].clone()
     for j,q in enumerate(public):
      rec=evidence.records[(q.known_entity_id,q.oriented_relation_id,q.timestamp)]
      if rec.get('abstain'):mask[j]=False
     sem=torch.stack([feat[n] for n in ['llm_confidence','llm_mapping_score','llm_template_agreement','llm_temporal_score','llm_rank_prior']],dim=-1)
     z=candidate_features(base,public,ids,mask,sem,ch);delta=cm(z.cpu(),mask.cpu(),cs).to(device)
     corrected=corrected_scores(base,ids,mask,delta)
     assert float(delta.abs().max())<=.350001
     assert torch.equal(corrected[~mask.any(dim=1)],base[~mask.any(dim=1)])
     gids=base.argsort(dim=1,descending=True,stable=True)[:,:10];gmask=torch.ones_like(gids,dtype=torch.bool)
     gz=candidate_features(base,public,gids,gmask,base.new_zeros((*gids.shape,5)),ch)
     gd=gm(gz.cpu(),gmask.cpu(),gs).to(device);gcorrected=corrected_scores(base,gids,gmask,gd)
     matrices=dict(graph=base,legacy=legacy,graph_control_transfer=gcorrected,cbrc_transfer=corrected)
     for j,f in enumerate(batch):
      s,r,target,t=f;truth=filters.get((s,r,t),set())|{target};key=':'.join(map(str,f))
      row=dict(fact=list(f),ranks={method:average_rank(score[j],target,truth) for method,score in matrices.items()})
      db.execute('INSERT INTO ranks VALUES (?,?)',(key,json.dumps(row)));done.add(key)
     db.commit();space();status('transfer_evaluation',seed=seed,completed=len(done),expected=2*COUNT,elapsed_seconds=time.time()-started)
   history.add_facts(all_rows)
   for f in all_rows:support_by_rel.setdefault(f[1],[]).append(f)
   ch.commit_timestamp(all_rows)
 assert len(done)==2*COUNT
 records=[json.loads(v) for v, in db.execute('SELECT payload FROM ranks ORDER BY key')];db.close()
 with (dest/'query_ranks.jsonl').open('w',encoding='utf-8') as f:
  for row in records:f.write(json.dumps(row)+'\n')
 result=dict(experiment=protocol['experiment'],K=K,seed=seed,queries=2*COUNT,methods={m:rank_metrics([r['ranks'][m] for r in records]) for m in METHODS},protocol_sha256=signature,cache_sha256=sha(p),target_domain_head_fitting=False,source_assets=asset,seconds_this_run=time.time()-started)
 write(end,result);del model;torch.cuda.empty_cache();return result
