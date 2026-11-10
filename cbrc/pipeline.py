from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import torch

from .adapter import sha256,implementation_sha256
from .core import CandidateCorrector,VARIANTS,SCALES,corrected_scores,sparse_objective,average_rank


def read_export(path):
    path=Path(path)
    meta=json.loads((path/'export.json').read_text(encoding='utf-8'))
    if meta.get('schema_version')!=1:
        raise ValueError('unsupported export schema')
    for shard in meta['shards']:
        if sha256(path/shard['path'])!=shard['sha256']:
            raise ValueError('export shard changed')
    return path,meta


def batches(path,meta,partition,variant,batch_size=128,seed=None):
    order=list(meta['shards'])
    rng=random.Random(seed)
    if seed is not None: rng.shuffle(order)
    for shard in order:
        data=torch.load(path/shard['path'],map_location='cpu',weights_only=True)
        ix=[i for i,r in enumerate(data['records']) if r['partition']==partition]
        if seed is not None: rng.shuffle(ix)
        for start in range(0,len(ix),batch_size):
            idx=ix[start:start+batch_size]
            if not idx: continue
            prefix='graph_' if variant=='graph_control' else ''
            yield dict(base=data['base'][idx],ids=data[prefix+'ids'][idx],mask=data[prefix+'mask'][idx],
                z=data[prefix+'z'][idx],fixed_delta=data['fixed_delta'][idx],
                records=[data['records'][i] for i in idx])


def ranks_for(batch,delta):
    scores=corrected_scores(batch['base'],batch['ids'],batch['mask'],delta)
    return [average_rank(s,r['target'],r['truths']) for s,r in zip(scores,batch['records'])]


def rank_metrics(ranks):
    if not ranks: raise ValueError('empty evaluation partition')
    n=len(ranks)
    return dict(mrr=sum(1/r for r in ranks)/n,**{f'hits{k}':sum(r<=k for r in ranks)/n for k in (1,3,10)},queries=n)


def assess(model,path,meta,partition,scale):
    ranks=[]
    with torch.no_grad():
        for batch in batches(path,meta,partition,model.variant):
            ranks.extend(ranks_for(batch,model(batch['z'],batch['mask'],scale)))
    return rank_metrics(ranks)


def fit(export_path,out,epochs=30,patience=5,variants=VARIANTS):
    if not 1<=epochs<=30 or patience!=5:
        raise ValueError('maximum 30 epochs, fixed patience 5')
    path,meta=read_export(export_path)
    if meta['split']!='valid': raise ValueError('fitting on test data is prohibited')
    if meta['cbrc_source_sha256']!=implementation_sha256():
        raise ValueError('re-export validation after changing the CBRC implementation')
    out=Path(out)
    if (out/'selection.json').exists(): raise FileExistsError('selection is frozen; use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    result={}
    for variant in variants:
        torch.manual_seed(meta['seed'])
        model=CandidateCorrector(variant)
        assert sum(p.numel() for p in model.parameters())==193
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best_mrr=assess(model,path,meta,'select',.35)['mrr']
        best=copy.deepcopy(model.state_dict()); best_epoch=0; stall=0; trace=[]
        for epoch in range(1,epochs+1):
            model.train(); losses=[]
            for batch in batches(path,meta,'fit',variant,seed=meta['seed']+epoch):
                optimizer.zero_grad(set_to_none=True)
                delta=model(batch['z'],batch['mask'],.35)
                loss,parts=sparse_objective(batch['base'],batch['ids'],batch['mask'],delta,
                    [r['truths'] for r in batch['records']],0. if variant=='no_kl' else 1.)
                if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                loss.backward(); optimizer.step(); losses.append(float(loss))
            if not losses: raise ValueError('no fit rows')
            model.eval(); metric=assess(model,path,meta,'select',.35)['mrr']
            trace.append(dict(epoch=epoch,loss=sum(losses)/len(losses),select_mrr=metric))
            if metric>best_mrr+1e-12:
                best_mrr=metric; best=copy.deepcopy(model.state_dict()); best_epoch=epoch; stall=0
            else: stall+=1
            if stall>=patience: break
        model.load_state_dict(best); model.eval()
        grid=[dict(scale=s,**assess(model,path,meta,'select',s)) for s in SCALES]
        chosen=max(grid,key=lambda m:(m['mrr'],-m['scale']))
        # Audit is diagnostic only; it cannot select a scale/epoch or veto individual seeds.
        audit=assess(model,path,meta,'audit',chosen['scale'])
        model_file=f'{variant}.pt'
        torch.save(dict(model=model.state_dict(),variant=variant,scale=chosen['scale'],
                        checkpoint_sha256=meta['checkpoint_sha256']),out/model_file)
        result[variant]=dict(model_file=model_file,sha256=sha256(out/model_file),
            scale=chosen['scale'],best_epoch=best_epoch,selection_grid=grid,
            development_audit=audit,trace=trace,parameter_count=193)
    selection=dict(schema_version=1,dataset=meta['dataset'],K=meta['K'],seed=meta['seed'],smoke=meta['smoke'],
        experiment_id=meta.get('experiment_id','original_block_b'),asset_origin=meta.get('asset_origin','verified_formal'),
        checkpoint_sha256=meta['checkpoint_sha256'],validation_cache_sha256=meta['cache_sha256'],
        dataset_fingerprint=meta['dataset_fingerprint'],source_model_sha256=meta['source_model_sha256'],
        cbrc_source_sha256=meta['cbrc_source_sha256'],
        valid_export_sha256=sha256(path/'export.json'),variants=result,
        development_audit_independent=False,supervision='validation calibration labels; graph frozen',
        training=dict(max_epochs=epochs,patience=5,lr=.001,weight_decay=.0001,kl_weight=1.,scales=list(SCALES)))
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    return selection


def evaluate(export_path,selection_path,out):
    path,meta=read_export(export_path)
    if meta['split']!='test': raise ValueError('use selection/audit scores for development')
    selection_path=Path(selection_path)
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if meta['selection_sha256']!=sha256(selection_path): raise ValueError('selection changed after test export')
    if selection['cbrc_source_sha256']!=implementation_sha256():
        raise ValueError('CBRC implementation changed after selection')
    for field in ('dataset','K','seed','smoke','checkpoint_sha256','dataset_fingerprint','source_model_sha256','cbrc_source_sha256'):
        if selection[field]!=meta[field]: raise ValueError(f'selection/test mismatch: {field}')
    out=Path(out)
    if (out/'evaluation.json').exists(): raise FileExistsError('test evaluation already frozen')
    out.mkdir(parents=True,exist_ok=True)
    all_ranks={}; times={}; details={}
    for method in ('graph','legacy','fixed_after',*selection['variants']):
        model=None
        variant=method if method in selection['variants'] else 'cbrc'
        if model_spec:=selection['variants'].get(method):
            model_path=selection_path.parent/model_spec['model_file']
            if sha256(model_path)!=model_spec['sha256']: raise ValueError('corrector checkpoint changed')
            model=CandidateCorrector(method)
            model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True)['model'])
            model.eval()
        ranks=[]; records=[]; elapsed=0.
        with torch.no_grad():
            for batch in batches(path,meta,'test',variant):
                tick=time.perf_counter()
                if method in ('graph','legacy'):
                    rr=[r['base_rank' if method=='graph' else 'legacy_rank'] for r in batch['records']]
                else:
                    delta=batch['fixed_delta'] if method=='fixed_after' else model(batch['z'],batch['mask'],model_spec['scale'])
                    # Correction timing excludes file reads and ranking; not an end-to-end speed claim.
                    elapsed+=time.perf_counter()-tick
                    rr=ranks_for(batch,delta)
                for r,rank in zip(batch['records'],rr):
                    change=1/rank-1/r['base_rank']
                    records.append(dict(**r['public_query'],target=r['target'],rank=rank,
                                        delta_rr_vs_graph=change,base_rank=r['base_rank']))
                ranks.extend(rr)
        all_ranks[method]=ranks; times[method]=elapsed
        positive=sum(max(r['delta_rr_vs_graph'],0.) for r in records)/len(records)
        negative=sum(min(r['delta_rr_vs_graph'],0.) for r in records)/len(records)
        details[method]=dict(**rank_metrics(ranks),improved=sum(r['delta_rr_vs_graph']>0 for r in records),
            unchanged=sum(r['delta_rr_vs_graph']==0 for r in records),damaged=sum(r['delta_rr_vs_graph']<0 for r in records),
            positive_rr_contribution=positive,negative_rr_contribution=negative,
            correction_seconds_on_exported_scores=elapsed)
        with (out/f'{method}_queries.jsonl').open('w',encoding='utf-8') as f:
            for r in records: f.write(json.dumps(r)+'\n')
    result=dict(dataset=meta['dataset'],K=meta['K'],seed=meta['seed'],smoke=meta['smoke'],
        experiment_id=meta.get('experiment_id','original_block_b'),asset_origin=meta.get('asset_origin','verified_formal'),
        checkpoint_sha256=meta['checkpoint_sha256'],selection_sha256=sha256(selection_path),
        test_export_sha256=sha256(path/'export.json'),queries=meta['queries'],methods=details,
        delta_mrr_cbrc_minus_legacy=details['cbrc']['mrr']-details['legacy']['mrr'],
        baseline_seconds=meta['baseline_seconds'],legacy_seconds=meta['legacy_seconds'],
        cost_scope='base forward on device separately; head on exported CPU scores; no matched end-to-end throughput claim')
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
