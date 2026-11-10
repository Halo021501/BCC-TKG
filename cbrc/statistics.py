from __future__ import annotations

import csv
import itertools
import json
import random
import statistics
from pathlib import Path


def exact_signed_rank(differences):
    ds=[d for d in differences if abs(d)>1e-12]
    if not ds: return 1.
    a=[round(abs(d),12) for d in ds]
    ranks=[1+sum(x<v for x in a)+(sum(x==v for x in a)-1)/2 for v in a]
    total=sum(ranks); obs=min(sum(r for r,d in zip(ranks,ds) if d>0),sum(r for r,d in zip(ranks,ds) if d<0))
    tail=0
    for bits in itertools.product((0,1),repeat=len(ds)):
        val=sum(r*b for r,b in zip(ranks,bits))
        tail+=min(val,total-val)<=obs+1e-10
    return tail/(2**len(ds))


def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); adjusted=[1.]*len(values); prev=0.
    for j,i in enumerate(order):
        prev=max(prev,min(1.,values[i]*(len(values)-j))); adjusted[i]=prev
    return adjusted


def paired_statistics(ds,seed=20261019):
    rng=random.Random(seed); n=len(ds)
    samples=sorted(sum(rng.choices(ds,k=n))/n*100 for _ in range(10000))
    def percentile(q):
        pos=(len(samples)-1)*q; lo=int(pos); fraction=pos-lo
        return samples[lo]*(1-fraction)+samples[min(lo+1,len(samples)-1)]*fraction
    return dict(mean_delta_pp=statistics.mean(ds)*100,ci95_low_pp=percentile(.025),
        ci95_high_pp=percentile(.975),positive=sum(d>0 for d in ds),negative=sum(d<0 for d in ds),
        p_exact=exact_signed_rank(ds),bootstrap_resamples=10000,bootstrap_seed=seed)


def summarize(result_paths,out):
    cells={}
    for p in result_paths:
        result=json.loads(Path(p).read_text(encoding='utf-8'))
        if result['smoke']: raise ValueError('smoke runs are excluded from formal inference')
        key=(result['dataset'],result['K'],result['seed'])
        if key in cells: raise ValueError('duplicate condition/seed')
        expected_queries={'ICEWS14':14742,'YAGO':40052}.get(result['dataset'])
        if result.get('queries')!=expected_queries:
            raise ValueError('incomplete or unknown processed test split')
        required={'graph','legacy','fixed_after','cbrc','graph_control','positive','no_semantic','no_graph','no_kl'}
        if set(result['methods'])!=required or any(m['queries']!=expected_queries for m in result['methods'].values()):
            raise ValueError('all nine registered methods must cover the identical complete query set')
        delta=result['methods']['cbrc']['mrr']-result['methods']['legacy']['mrr']
        if abs(delta-result['delta_mrr_cbrc_minus_legacy'])>1e-12:
            raise ValueError('stored contrast disagrees with method metrics')
        cells[key]=result
    expected={(d,k,s) for d in ('ICEWS14','YAGO') for k in (5,10) for s in range(42,52)}
    missing=sorted(expected-cells.keys()); extra=sorted(cells.keys()-expected)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if missing or extra:
        report=dict(status='incomplete',missing=missing,extra=extra,adopt_module=False,
                    reason='All forty registered test cells required; no selective seed deletion.')
        (out/'decision.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report
    rows=[]
    for i,(d,k) in enumerate(itertools.product(('ICEWS14','YAGO'),(5,10))):
        ds=[cells[d,k,s]['delta_mrr_cbrc_minus_legacy'] for s in range(42,52)]
        rows.append(dict(dataset=d,K=k,**paired_statistics(ds,20261019+i)))
    adjusted=holm([r['p_exact'] for r in rows])
    for r,p in zip(rows,adjusted): r['p_holm']=p
    adopt=all(r['mean_delta_pp']>0 for r in rows) and sum(r['p_holm']<.05 and r['mean_delta_pp']>0 for r in rows)>=2
    with (out/'primary_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    performance=[]
    for d,k in itertools.product(('ICEWS14','YAGO'),(5,10)):
        for method in sorted(cells[d,k,42]['methods']):
            row=dict(dataset=d,K=k,method=method,n=10)
            for metric in ('mrr','hits1','hits3','hits10','improved','unchanged','damaged',
                           'positive_rr_contribution','negative_rr_contribution','correction_seconds_on_exported_scores'):
                values=[cells[d,k,s]['methods'][method][metric] for s in range(42,52)]
                row[metric+'_mean']=statistics.mean(values);row[metric+'_sample_sd']=statistics.stdev(values)
            performance.append(row)
    with (out/'performance_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(performance[0]));w.writeheader();w.writerows(performance)
    report=dict(status='complete',adopt_module=adopt,primary_contrasts=rows,
        interpretation='cross-setting improvement criterion met' if adopt else 'no general improvement claim; report all cells',
        inference_scope='graph seeds conditional on fixed datasets and caches; four-test Holm family; intervals not simultaneous')
    (out/'decision.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
