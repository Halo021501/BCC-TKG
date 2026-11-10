from __future__ import annotations

import bisect
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from .adapter import load_frozen, sha256, verified_assets, implementation_sha256
from .core import PublicQuery,CausalCandidateHistory,candidate_features,average_rank,chronological_partitions


def export_split(spec,code_root,split,out,device='cpu',batch_size=64,smoke=False,selection=None,limit=0):
    out=Path(out)
    if (out/'export.json').exists() or list(out.glob('shard-*.pt')):
        raise FileExistsError('use a fresh export directory')
    if limit and not smoke:
        raise ValueError('partial data are permitted only in explicitly labeled smoke runs')
    paths,identities=verified_assets(spec,code_root,split,smoke)
    if split=='test':
        if not selection:
            raise ValueError('freeze the fitted selection.json before exporting test scores')
        selected=json.loads(Path(selection).read_text(encoding='utf-8'))
        if selected['checkpoint_sha256']!=identities['checkpoint']:
            raise ValueError('selection uses a different checkpoint')
        if selected['smoke']!=smoke:
            raise ValueError('smoke/formal selection mismatch')
        if selected['cbrc_source_sha256']!=implementation_sha256():
            raise ValueError('CBRC implementation changed after selection was frozen')
    sys.path.insert(0,str(Path(code_root).resolve()))
    data=importlib.import_module('src.data'); train=importlib.import_module('src.train')
    llm=importlib.import_module('src.llm_cache')
    torch.set_num_threads(1); train.set_seed(spec['seed'])
    kg=data.load_temporal_kg(str(paths['data_dir']))
    if not smoke:
        if spec['dataset']=='ICEWS14' and (len(kg.train),len(kg.valid),len(kg.test))!=(74845,8514,7371):
            raise ValueError('wrong ICEWS14 processed split')
        if spec['dataset']=='YAGO' and (len(kg.train),len(kg.valid),len(kg.test))!=(161540,19523,20026):
            raise ValueError('wrong YAGO processed split')
    model,cfg=load_frozen(code_root,paths['checkpoint'],kg,torch.device(device))
    if cfg['shot']!=spec['K'] or cfg['seed']!=spec['seed']:
        raise ValueError('checkpoint K/seed mismatch')
    cache=llm.LLMEvidenceCache(str(paths[split+'_cache']),max_candidates=10,expected_shot=spec['K'],
        expected_history_protocol='standard_rolling_history',expected_split=split,
        expected_dataset_fingerprint=llm.dataset_files_fingerprint(str(paths['data_dir'])),
        require_generation_metadata=not smoke)
    initial=kg.train_aug+(kg.valid_aug if split=='test' else [])
    rows=list(kg.test if split=='test' else kg.valid)
    if limit: rows=rows[:limit]
    if max(f[3] for f in initial)>=min(f[3] for f in rows):
        raise ValueError('initial history overlaps evaluation timestamps; causal protocol needs explicit resolution')
    history=data.HistoryIndex(initial,kg.num_entities,kg.num_relations*2,cfg['history_len'],
        adaptive_decay=not cfg.get('disable_temporal_calibration',False))
    support_by_rel=train.group_by_relation(initial)
    ch=CausalCandidateHistory(kg.train_aug)
    for s,r,v,t in (kg.valid_aug if split=='test' else []):
        bisect.insort(ch.times[s,r,v],t)
    # Only already known validation labels define calibration multi-positive loss.
    filters=data.build_filter_dict(data.add_inverse(kg.train+kg.valid+(kg.test if split=='test' else []),kg.num_relations))
    parts=chronological_partitions([q[3] for q in rows]) if split=='valid' else {}
    snapshots=defaultdict(list)
    for q in rows: snapshots[q[3]].append(q)
    out.mkdir(parents=True,exist_ok=True)
    shards=[]; total=0; hits=0; elapsed_base=0.; elapsed_legacy=0.
    def sync():
        if str(device).startswith('cuda'): torch.cuda.synchronize()
    with torch.inference_mode():
        for timestamp in sorted(snapshots):
            ch.begin_timestamp(timestamp)
            oriented=data.add_inverse(snapshots[timestamp],kg.num_relations)
            grouped=train.group_by_relation(oriented)
            for relation in sorted(grouped):
                rel_rows=grouped[relation]
                for start in range(0,len(rel_rows),batch_size):
                    batch=rel_rows[start:start+batch_size]
                    public=[PublicQuery(s,r,t) for s,r,_,t in batch]
                    sanitized=[(q.known_entity_id,q.oriented_relation_id,0,q.timestamp) for q in public]
                    supports=train.causal_support_rows(support_by_rel,sanitized,spec['K'])
                    if any(f[3]>=timestamp for sup in supports for f in sup):
                        raise ValueError('noncausal support (including invalid structural sentinel)')
                    st=train.support_rows_tensor(supports,torch.device(device))
                    features=train.move_features(cache.augment_features(sanitized,history.build(sanitized)),torch.device(device))
                    qt=data.tensorize(sanitized,torch.device(device))
                    hits+=int(features['llm_cache_hit'].sum())
                    model.set_llm_runtime_mode('off'); sync(); tick=time.perf_counter()
                    base,aux=model(qt,st,features); sync(); elapsed_base+=time.perf_counter()-tick
                    bank=aux['alterego_candidate_ids'].clone()
                    model.set_llm_runtime_mode('rationale'); sync(); tick=time.perf_counter()
                    legacy,_=model(qt,st,features); sync(); elapsed_legacy+=time.perf_counter()-tick
                    model.set_llm_runtime_mode('off')
                    # Explicit masks incorporate cache abstention even if a malformed producer kept candidates.
                    ids=features['llm_candidate_ids']; mask=features['llm_candidate_mask'].clone()
                    for i,q in enumerate(public):
                        rec=cache.records.get((q.known_entity_id,q.oriented_relation_id,q.timestamp),{})
                        if rec.get('abstain',False) or rec.get('abstained',False): mask[i]=False
                    sem=torch.stack([features[n] for n in ('llm_confidence','llm_mapping_score',
                        'llm_template_agreement','llm_temporal_score','llm_rank_prior')],dim=-1)
                    z=candidate_features(base,public,ids,mask,sem,ch)
                    graph_ids=base.argsort(dim=1,descending=True,stable=True)[:,:min(10,kg.num_entities)]
                    graph_mask=torch.ones_like(graph_ids,dtype=torch.bool)
                    gz=candidate_features(base,public,graph_ids,graph_mask,base.new_zeros((*graph_ids.shape,5)),ch)
                    records=[]
                    for i,(s,r,target,t) in enumerate(batch):
                        truth=sorted(filters.get((s,r,t),{target})|{target})
                        records.append(dict(public_query=public[i].__dict__,target=target,truths=truth,
                            partition=parts[t] if split=='valid' else 'test',
                            base_rank=average_rank(base[i],target,truth),legacy_rank=average_rank(legacy[i],target,truth)))
                    payload=dict(base=base.cpu(),ids=ids.cpu(),mask=mask.cpu(),z=z.cpu(),
                        graph_ids=graph_ids.cpu(),graph_mask=graph_mask.cpu(),graph_z=gz.cpu(),
                        graph_bank=bank.cpu(),
                        records=records)
                    # Reproduce the original fixed semantic sidecar as an independent post-tournament control.
                    # The frozen helper uses its exact original feature weights and bound.
                    model.set_llm_runtime_mode('rationale')
                    _,bonus,*_=model._apply_llm_side_evidence(base,features)
                    model.set_llm_runtime_mode('off')
                    payload['fixed_delta']=bonus.gather(1,ids).cpu()*mask.cpu()
                    name=f'shard-{len(shards):06d}.pt'
                    torch.save(payload,out/name)
                    shards.append(dict(path=name,sha256=sha256(out/name),queries=len(batch)))
                    total+=len(batch)
            history.add_facts(oriented)
            for f in oriented: support_by_rel.setdefault(f[1],[]).append(f)
            ch.commit_timestamp(oriented)
    if hits!=total and not smoke:
        raise ValueError(f'formal cache incomplete: {hits}/{total}; no silent partial formal evaluation')
    metadata=dict(schema_version=1,dataset=spec['dataset'],K=spec['K'],seed=spec['seed'],split=split,
        experiment_id=spec.get('experiment_id','original_block_b'),asset_origin=spec['status'],
        smoke=smoke,limited=bool(limit),queries=total,cache_hits=hits,entities=kg.num_entities,
        checkpoint_sha256=identities['checkpoint'],cache_sha256=identities[split+'_cache'],
        dataset_fingerprint=llm.dataset_files_fingerprint(str(paths['data_dir'])),
        source_model_sha256=sha256(Path(code_root)/'src/model.py'),
        cbrc_source_sha256=implementation_sha256(),
        baseline_seconds=elapsed_base,legacy_seconds=elapsed_legacy,shards=shards,
        selection_sha256=sha256(selection) if selection else None,
        protocol='observed feedback, complete timestamp barrier, target-blind input, average-tie filtering',
        development_audit_independent=False)
    (out/'export.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata
