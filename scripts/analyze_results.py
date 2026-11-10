"""Recompute numeric results and paired tests; never creates manuscript text."""
from _common import ROOT,read,write
from pathlib import Path
import argparse
import csv
import itertools
import random
import statistics as st
import numpy as np
from cbrc.statistics import exact_signed_rank,holm,paired_statistics

METRICS=('mrr','hits1','hits3','hits10')
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def signflip(ds):
    obs=abs(sum(ds));n=len(ds)
    return sum(abs(sum(d*s for d,s in zip(ds,ss)))>=obs-1e-12
        for ss in itertools.product((-1,1),repeat=n))/2**n
def summary(raw,scope):
    groups={}
    for r in raw:
        key=(int(r['K']),r['method']);seed=int(r['seed'])
        if seed in groups.setdefault(key,{}):raise ValueError('Duplicate seed in a method')
        groups[key][seed]=r
    result=[]
    for (k,m),rr in sorted(groups.items()):
        if set(rr)!=set(range(42,52)):raise ValueError('Expected all ten seeds, no deletion or replacement')
        rec=dict(scope=scope,K=k,method=m,n=10)
        for metric in METRICS:
            v=[100*float(rr[s][metric]) for s in range(42,52)]
            rec[metric+'_mean_pct']=st.mean(v);rec[metric+'_sample_sd_pct']=st.stdev(v)
        result.append(rec)
    return result,groups
def effect(groups,k,method,ref,kind):
    ds=[float(groups[k,method][s]['mrr'])-float(groups[k,ref][s]['mrr']) for s in range(42,52)]
    if kind=='icews':
        pp=np.array(ds)*100;rng=np.random.default_rng(20261024)
        bs=pp[rng.integers(0,10,size=(100000,10))].mean(1)
        lo,hi=np.percentile(bs,[2.5,97.5]);p=signflip(pp);count=100000
    else:
        rng=random.Random(20261028)
        bs=sorted(100*st.mean(rng.choices(ds,k=10)) for _ in range(10000))
        lo,hi=bs[249],bs[9749];p=signflip(ds);count=10000
    return dict(scope='ICEWS14 complete-test correction' if kind=='icews' else 'YAGO 4000-instance transfer',
        K=k,method=method,reference=ref,delta_mrr_pp=100*st.mean(ds),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
        wins=sum(d>0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d<0 for d in ds),
        p_exact=p,p_holm='',test='two-sided mean sign-flip',bootstrap_resamples=count)
def analyze(reference,out):
    reference=Path(reference);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    a,ig=summary(rows(reference/'icews14_correction.csv'),'ICEWS14 complete-test correction')
    b,tg=summary(rows(reference/'yago_transfer.csv'),'YAGO 4000-instance transfer')
    c=rows(reference/'complete_test_semantic_pairs.csv')
    c=[r for r in c if r['gate']=='gateoff']
    if len(c)!=40:raise ValueError('Expected four ten-seed complete-test comparisons')
    complete=[];contrasts=[]
    for index,(dataset,k) in enumerate(itertools.product(('ICEWS14','YAGO'),(5,10))):
        rr=sorted([r for r in c if r['dataset']==dataset and int(r['K'])==k],key=lambda r:int(r['seed']))
        if [int(r['seed']) for r in rr]!=list(range(42,52)):raise ValueError('Missing complete-test seeds')
        ds=[float(r['rationale_mrr'])-float(r['off_mrr']) for r in rr]
        values=np.asarray(ds)
        bs=values[np.random.default_rng(20260909+index).integers(0,10,size=(10000,10))].mean(1)*100
        low,high=np.percentile(bs,[2.5,97.5])
        contrasts.append(dict(scope=dataset+' complete-test PSCF',K=k,method='pscf',reference='graph',
            delta_mrr_pp=100*st.mean(ds),ci95_low_pp=float(low),ci95_high_pp=float(high),
            wins=sum(d>0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d<0 for d in ds),
            p_exact=exact_signed_rank(ds),p_holm='',test='two-sided signed-rank',bootstrap_resamples=10000))
        for method,prefix in [('graph','off'),('pscf','rationale')]:
            rec=dict(scope=dataset+' complete-test PSCF',K=k,method=method,n=10)
            for metric in METRICS:
                v=[100*float(r[prefix+'_'+metric]) for r in rr]
                rec[metric+'_mean_pct']=st.mean(v);rec[metric+'_sample_sd_pct']=st.stdev(v)
            complete.append(rec)
    for r,p in zip(contrasts,holm([r['p_exact'] for r in contrasts])):r['p_holm']=p
    corrections=[effect(ig,k,'cbrc',ref,'icews') for k in (5,10)
                 for ref in ('graph','legacy','fixed_after','graph_control','positive','no_semantic','no_graph','no_kl')]
    primary=[r for r in corrections if r['reference']=='legacy']
    # Preserve the original four-comparison family; two unrun full-YAGO CBRC cells have p=1.
    adjusted=holm([r['p_exact'] for r in primary]+[1.,1.])
    for r,p in zip(primary,adjusted):r['p_holm']=p
    transfer=[effect(tg,5,'cbrc_transfer',ref,'transfer') for ref in ('graph','legacy','graph_control_transfer')]
    csvout(out/'performance_summary.csv',complete+a+b)
    csvout(out/'paired_comparisons.csv',contrasts+corrections+transfer)
    write(out/'analysis_scope.json',dict(complete_test_conditions=4,icews_correction_conditions=2,
        transfer_conditions=1,transfer_instances_per_seed=4000,seeds=list(range(42,52)),
        method_labels={'legacy':'PSCF','cbrc':'CBRC','graph_control':'graph-only corrector'},
        full_yago_cbrc_result_available=False,conditional_on_fixed_evidence=True,
        multiplicity='Four full-test PSCF comparisons; separate original four-comparison CBRC/PSCF family with unrun p=1 placeholders. Other contrasts exploratory.'))
    print('Numeric analysis:',out)

def collect(paths,out):
    """Collect new evaluation.json files without mixing scopes or populations."""
    rr=[]
    for path in paths:
        result=read(path)
        if result.get('smoke'):raise ValueError('Synthetic smoke results cannot enter paper summaries')
        scope='transfer' if result.get('target_domain_head_fitting') is False else result.get('scope','correction')
        expected=4000 if scope=='transfer' else (14742 if result.get('dataset')=='ICEWS14' else 40052)
        if result['queries']!=expected:raise ValueError('Unexpected evaluation population')
        for method,metrics in result['methods'].items():
            row=dict(scope=scope,dataset=result.get('dataset','YAGO'),K=result['K'],seed=result['seed'],method=method,queries=result['queries'])
            row.update({m:metrics[m] for m in METRICS});rr.append(row)
    keys=[(r['scope'],r['dataset'],r['K'],r['seed'],r['method']) for r in rr]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate run keys')
    csvout(Path(out)/'collected_seed_metrics.csv',rr)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,default=ROOT/'results_reference')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evaluations',nargs='+',type=Path,help='Collect fresh run metrics instead of analyzing bundled reference data')
    a=p.parse_args()
    if a.evaluations:collect(a.evaluations,a.out)
    else:analyze(a.reference_dir,a.out)
