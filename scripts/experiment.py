"""Portable entry points for the recorded full-data graph and CBRC experiments."""
from _common import ROOT, CODE, read, write, sha, verify_sources, lock
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

def check_data(dataset, directory):
    from src.data import load_temporal_kg
    from src.llm_cache import dataset_files_fingerprint
    directory=Path(directory).resolve()
    hashes=read(ROOT/'configs/dataset_checksums.json')[dataset]
    for name, digest in hashes.items():
        if sha(directory/name)!=digest: raise ValueError('Dataset bytes differ: '+name)
    kg=load_temporal_kg(str(directory))
    expected={'ICEWS14':(74845,8514,7371,7128,230,365),
              'YAGO':(161540,19523,20026,10623,10,189)}[dataset]
    actual=(len(kg.train),len(kg.valid),len(kg.test),kg.num_entities,kg.num_relations,kg.num_times)
    if actual!=expected: raise ValueError(f'Dataset shape mismatch: {actual}, expected {expected}')
    return dict(dataset=dataset,counts=actual,fingerprint=dataset_files_fingerprint(str(directory)))

def train(args):
    verify_sources(); check_data(args.dataset,args.data_dir)
    out=args.out.resolve(); graph=out/'graph'
    config=read(ROOT/f'configs/{args.dataset.lower()}_k{args.K}.json')
    values={'{DATA}':str(args.data_dir.resolve()),'{OUTPUT}':str(graph),
            '{RESOURCE}':str(graph/'resources.csv'),'{QUERIES}':str(graph/'query_diagnostics')}
    argv=[values.get(x,x) for x in config['arguments']]
    for flag,value in [('--seed',str(args.seed)),('--device',args.device)]:
        argv[argv.index(flag)+1]=value
    argv+=['--source-manifest',str(CODE/'provenance/source_manifest.original.json')]
    command=[sys.executable,'-u',str(ROOT/'runtime/resumable_train.py'),*argv]
    if args.dry_run: print(json.dumps(command,indent=2)); return
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; refusing a silent CPU fallback')
    with lock(out):
        plan=out/'training_plan.json'
        if plan.exists() and read(plan)['command']!=command: raise ValueError('Training plan changed')
        write(plan,dict(command=command,dataset=args.dataset,K=args.K,seed=args.seed))
        marker=out/'graph_complete.json'
        if marker.exists():
            if sha(graph/'best.pt')!=read(marker)['best_sha256']: raise ValueError('Completed checkpoint changed')
            print('Already complete:',marker); return
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONUTF8':'1',
             'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CBRC_CHECKPOINT_SECONDS':'120'}
        for key in list(env):
            if key.startswith('CBRC_TEST_'): env.pop(key)
        with (out/'train.log').open('a',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Training stopped ({result.returncode}); rerun the identical command. See {out / "train.log"}')
        write(marker,dict(complete=True,best_sha256=sha(graph/'best.pt'),metrics_sha256=sha(graph/'metrics.json')))
        print('Completed:',marker)

def calibrate(args):
    from cbrc.export import export_split
    from cbrc.pipeline import evaluate
    from resumable_fit import fit
    verify_sources(); check_data(args.dataset,args.data_dir)
    out=args.out.resolve()
    spec=dict(dataset=args.dataset,K=args.K,seed=args.seed,
              checkpoint=str(args.checkpoint.resolve()),data_dir=str(args.data_dir.resolve()),
              valid_cache=str(args.valid_cache.resolve()),test_cache=str(args.test_cache.resolve()),
              status='verified_new_experiment',experiment_id=f'{args.dataset}-K{args.K}-seed{args.seed}',
              recovery_provenance='User-supplied graph checkpoint and immutable evidence; training plan and hashes retained.',
              dataset_sha256=read(ROOT/'configs/dataset_checksums.json')[args.dataset],
              sha256=dict(checkpoint=sha(args.checkpoint),valid_cache=sha(args.valid_cache),test_cache=sha(args.test_cache)))
    with lock(out):
        assets=out/'assets.json'
        if assets.exists() and read(assets)['conditions']!=[spec]: raise ValueError('Assets changed')
        write(assets,dict(code_root=str(CODE),conditions=[spec]))
        for split in ('valid','test'):
            if split=='test':
                fit(out/'valid',out/'fit')
            dest=out/split
            if not (dest/'export.json').exists():
                if list(dest.glob('shard-*.pt')):
                    raise RuntimeError(f'Incomplete dense export at {dest}. Preserve it and use a new calibration output directory; no files are deleted automatically.')
                export_split(spec,CODE,split,dest,args.device,args.batch_size,
                             selection=out/'fit/selection.json' if split=='test' else None)
        if not (out/'evaluation/evaluation.json').exists():
            evaluate(out/'test',out/'fit/selection.json',out/'evaluation')
        print('Evaluation:',out/'evaluation/evaluation.json')

def paired(args):
    """Whole-test graph vs PSCF on one identical, frozen checkpoint."""
    import torch
    from cbrc.adapter import load_frozen
    from src.data import load_temporal_kg,HistoryIndex,build_filter_dict
    from src.llm_cache import LLMEvidenceCache,dataset_files_fingerprint,cache_coverage
    from src.train import evaluate,group_by_relation,set_seed
    verify_sources(); check_data(args.dataset,args.data_dir)
    out=args.out.resolve()
    with lock(out):
        if (out/'evaluation.json').exists(): raise FileExistsError('Evaluation already frozen')
        kg=load_temporal_kg(str(args.data_dir)); device=torch.device(args.device)
        torch.set_num_threads(1); set_seed(args.seed)
        model,cfg=load_frozen(CODE,args.checkpoint,kg,device)
        if (cfg['shot'],cfg['seed'])!=(args.K,args.seed): raise ValueError('Checkpoint K/seed mismatch')
        evidence=LLMEvidenceCache(str(args.cache),max_candidates=10,expected_shot=args.K,expected_split='test',
            expected_dataset_fingerprint=dataset_files_fingerprint(str(args.data_dir)),
            expected_history_protocol='standard_rolling_history',require_generation_metadata=True)
        coverage=cache_coverage(evidence,kg.test_aug)
        if coverage['cache_hit_rate']!=1: raise ValueError('Incomplete test evidence cache')
        initial=kg.train_aug+kg.valid_aug
        filters=build_filter_dict(kg.train_aug+kg.valid_aug+kg.test_aug)
        metrics={}
        with torch.inference_mode():
            for name,mode in [('graph','off'),('pscf','rationale')]:
                model.set_llm_runtime_mode(mode)
                history=HistoryIndex(initial,kg.num_entities,kg.num_relations*2,cfg['history_len'],
                                     adaptive_decay=not cfg.get('disable_temporal_calibration',False))
                metrics[name]=evaluate(model,kg.test,history,group_by_relation(initial),kg.train_aug,filters,
                    kg.num_relations,args.K,device,eval_batch_size=args.batch_size,
                    history_protocol='standard_rolling_history',llm_cache=evidence,
                    query_export_path=str(out/f'{name}_queries.jsonl'))
        write(out/'evaluation.json',dict(dataset=args.dataset,K=args.K,seed=args.seed,
            checkpoint_sha256=sha(args.checkpoint),cache_sha256=sha(args.cache),
            scope='complete_test',queries=2*len(kg.test),methods=metrics,
            metric_note='Frozen trainer reports its filtered ranking and explicit average-tie diagnostics; compare like-named metrics.'))
        print('Evaluation:',out/'evaluation.json')

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('check');a.add_argument('--dataset',choices=['ICEWS14','YAGO'],required=True);a.add_argument('--data-dir',type=Path,required=True)
    for name in ('train','calibrate','paired'):
        a=sub.add_parser(name);a.add_argument('--dataset',choices=['ICEWS14','YAGO'],required=True)
        a.add_argument('--data-dir',type=Path,required=True);a.add_argument('--K',type=int,choices=[5,10],required=True)
        a.add_argument('--seed',type=int,choices=range(42,52),required=True);a.add_argument('--out',type=Path,required=True)
        a.add_argument('--device',default='cuda:0')
        if name=='train': a.add_argument('--dry-run',action='store_true')
        else:
            a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--batch-size',type=int,default=64)
            if name=='calibrate':
                a.add_argument('--valid-cache',type=Path,required=True);a.add_argument('--test-cache',type=Path,required=True)
            else: a.add_argument('--cache',type=Path,required=True)
    args=p.parse_args()
    if args.command=='check': verify_sources();print(json.dumps(check_data(args.dataset,args.data_dir),indent=2))
    else: {'train':train,'calibrate':calibrate,'paired':paired}[args.command](args)

if __name__=='__main__': main()
