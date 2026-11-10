"""Inventory local assets without treating similarly named runs as paper weights."""
from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path

import torch

from .adapter import sha256


def inventory(workspace,out):
    root=Path(workspace).resolve(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    references=root/'draft/Scientific_Reports_LaTeX_final_20261018/source_data'
    with (references/'block_b_dataset_checksums.csv').open(encoding='utf-8-sig') as f:
        expected=list(csv.DictReader(f))
    files=[]
    for parent,dirs,names in os.walk(root):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','node_modules','.venv','tectonic-cache','render_check','CBRC'}]
        for n in names:
            if n.endswith(('.pt','.pth','.zip')) or n in ('train.txt','valid.txt','test.txt','train','valid','test') or ('s5.jsonl' in n or 's10.jsonl' in n):
                files.append(Path(parent)/n)
    data=[]
    for directory in sorted({p.parent for p in files if p.name in ('train','train.txt')}):
        found={}
        for split in ('train','valid','test'):
            p=next((directory/n for n in (split+'.txt',split) if (directory/n).is_file()),None)
            if p: found[split]=dict(path=str(p),sha256=sha256(p))
        matched=[d for d in ('ICEWS14','YAGO') if len(found)==3 and all(
            found[r['split']]['sha256']==r['sha256'] for r in expected if r['dataset']==d)]
        if matched: data.append(dict(dataset=matched[0],directory=str(directory),splits=found))
    caches=[]
    for p in files:
        if not p.name.endswith('.jsonl') or not ('s5.' in p.name or 's10.' in p.name): continue
        try:
            with p.open(encoding='utf-8') as f: record=json.loads(next(f))
            q=record.get('query',{})
            if not {'split','shot','dataset_fingerprint'}<=q.keys(): continue
            caches.append(dict(path=str(p),sha256=sha256(p),split=q['split'],K=int(q['shot']),
                dataset_fingerprint=q['dataset_fingerprint'],metadata_exists=Path(str(p)+'.meta.json').is_file(),
                debug_marker=any(s in str(p).lower() for s in ('smoke','debug','toy','quick','probe','loader_cache')),
                metadata_path=str(p)+'.meta.json'))
        except (ValueError,StopIteration,UnicodeError): continue
    weights=[]
    for p in files:
        if p.suffix not in ('.pt','.pth'): continue
        info=dict(path=str(p),bytes=p.stat().st_size,formal_verified=False)
        try:
            cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            cfg=cp.get('args',{}) if isinstance(cp,dict) else {}
            state=cp.get('model',{}) if isinstance(cp,dict) else {}
            ent=state.get('entity.weight')
            info.update(seed=cfg.get('seed'),K=cfg.get('shot'),ungated=cfg.get('disable_support_gate',False),
                entity_shape=list(ent.shape) if ent is not None else None,
                has_sidecar='llm_sidecar.feature_logits' in state,
                has_tournament=any(k.startswith('alterego_tournament.') for k in state),
                debug_marker=any(s in str(p).lower() for s in ('smoke','debug','toy','quick','probe')),
                reason='No original Block B checkpoint identity/recovery provenance established')
            del cp,state
        except Exception as e:
            info['inspection_error']=str(e)[:300]
        weights.append(info)
    archive_hits=[]
    for p in files:
        if p.suffix!='.zip': continue
        try:
            with zipfile.ZipFile(p) as z:
                names=[n for n in z.namelist() if n.endswith(('.pt','.pth')) or ('yago' in n.lower() and n.endswith('.jsonl'))]
            if names: archive_hits.append(dict(path=str(p),members=names))
        except zipfile.BadZipFile: pass
    # Compute the exact cache fingerprint contract (split name plus raw bytes).
    from hashlib import sha256 as digest
    for d in data:
        h=digest()
        for split in ('train','valid','test'):
            h.update(split.encode())
            with Path(d['splits'][split]['path']).open('rb') as f:
                for block in iter(lambda:f.read(1<<20),b''): h.update(block)
        d['fingerprint']=h.hexdigest()
    grid=[]
    for dataset in ('ICEWS14','YAGO'):
        ds=next((d for d in data if d['dataset']==dataset),None)
        for k in (5,10):
            valid=next((c for c in caches if ds and not c['debug_marker'] and c['metadata_exists'] and c['dataset_fingerprint']==ds['fingerprint'] and c['K']==k and c['split']=='valid'),None)
            test=next((c for c in caches if ds and not c['debug_marker'] and c['metadata_exists'] and c['dataset_fingerprint']==ds['fingerprint'] and c['K']==k and c['split']=='test'),None)
            for seed in range(42,52):
                grid.append(dict(dataset=dataset,K=k,seed=seed,status='missing_original_checkpoint',
                    checkpoint='',data_dir=ds['directory'] if ds else '',valid_cache=valid['path'] if valid else '',
                    test_cache=test['path'] if test else '',sha256={x:c['sha256'] for x,c in [('valid_cache',valid),('test_cache',test)] if c},
                    dataset_sha256={Path(r['path']).name:r['sha256'] for r in ds['splits'].values()} if ds else {},
                    recovery_provenance=''))
    report=dict(matched_datasets=data,caches=caches,checkpoint_inventory=weights,archive_hits=archive_hits,
        verified_formal_checkpoints=0,formal_ready_conditions=0,
        note='Inventory candidates are not original paper weights. Recovery requires original identities and provenance; do not promote debug weights.')
    (out/'asset_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'assets_to_recover.json').write_text(json.dumps(dict(code_root=str(root/'代码/LLM-FSTKGC_20260908'),conditions=grid),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(dataset_matches=len(data),cache_files=len(caches),checkpoint_files=len(weights),
        archive_hits=len(archive_hits),formal_ready_conditions=0),indent=2))
    return report
