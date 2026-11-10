"""Apply source-domain correctors to a target-trained YAGO graph, with no refitting."""
from _common import ROOT,CODE,read,write,sha,verify_sources,lock
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from experiment import check_data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','checkpoint','source-selection','cache','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,choices=range(42,52),required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();verify_sources();check_data('YAGO',a.data_dir)
    import torch
    from src.data import load_temporal_kg
    from src.llm_cache import dataset_files_fingerprint
    from cbrc.adapter import implementation_sha256
    import _transfer_inference as t
    selection=read(a.source_selection)
    if (selection['dataset'],selection['K'],selection['seed'],selection['smoke'])!=('ICEWS14',5,a.seed,False):
        raise ValueError('Expected the same-number ICEWS14 K=5 source corrector')
    if selection['cbrc_source_sha256']!=implementation_sha256(): raise ValueError('Source corrector implementation changed')
    asset=dict(seed=a.seed,graph_checkpoint=str(a.checkpoint.resolve()),graph_sha256=sha(a.checkpoint),
               source_selection=str(a.source_selection.resolve()),selection_sha256=sha(a.source_selection),heads={})
    for variant in ('cbrc','graph_control'):
        row=selection['variants'][variant];file=a.source_selection.parent/row['model_file']
        if sha(file)!=row['sha256']:raise ValueError('Source head changed')
        asset['heads'][variant]=dict(path=str(file.resolve()),sha256=sha(file),scale=row['scale'])
    kg=load_temporal_kg(str(a.data_dir));subset=read(ROOT/'configs/yago_transfer_subset.json')
    if dataset_files_fingerprint(str(a.data_dir))!=subset['dataset_fingerprint']:raise ValueError('Subset refers to different processed data')
    selected={tuple(kg.test[i]) for i in subset['source_line_indices_zero_based']}
    if len(selected)!=2000:raise ValueError('Transfer selection is not 2000 distinct facts')
    protocol=dict(experiment='YAGO-fixed-query-calibrator-transfer-portable',K=5,
        dataset_fingerprint=subset['dataset_fingerprint'],queries=4000,assets=[asset],
        subset_sha256=sha(ROOT/'configs/yago_transfer_subset.json'),cache_sha256=sha(a.cache),
        target_domain_head_fitting=False,target_domain_scale_selection=False,
        cbrc_source_sha256=implementation_sha256(),inference_sha256=sha(Path(t.__file__)))
    out=a.out.resolve()
    with lock(out):
        if (out/'protocol.json').exists() and read(out/'protocol.json')!=protocol:raise ValueError('Transfer protocol changed')
        write(out/'protocol.json',protocol)
        t.OUT=out;t.CODE=CODE;t.DATA=a.data_dir.resolve();t.K=5;t.COUNT=2000;t.DEVICE=a.device
        t.METHODS=('graph','legacy','graph_control_transfer','cbrc_transfer')
        t.read=read;t.write=write;t.sha=sha
        def space():
            if shutil.disk_usage(out).free<4*1024**3:raise RuntimeError('Less than 4 GiB free; durable ranks retained')
        t.space=space;t.status=lambda stage,**kw:write(out/'status.json',dict(stage=stage,**kw))
        # load_frozen verifies the ungated target checkpoint; the source scales are unchanged.
        t.evaluate(kg,protocol,selected,a.cache.resolve(),asset)
        print('Evaluation:',out/f'seed{a.seed}/evaluation.json')

if __name__=='__main__':main()
