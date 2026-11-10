"""Fail closed on unavailable formal assets; otherwise print or execute all 40 cells."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from .adapter import verified_assets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.assets=a.assets.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.assets.read_text(encoding='utf-8'))
    lookup={(c['dataset'],c['K'],c['seed']):c for c in manifest['conditions']}
    expected=list(itertools.product(('ICEWS14','YAGO'),(5,10),range(42,52)))
    if len(lookup)!=40 or set(lookup)!=set(expected): raise ValueError('register exactly forty unique formal conditions')
    blockers=[]
    for key in expected:
        try:
            for split in ('valid','test'): verified_assets(lookup[key],manifest['code_root'],split)
        except Exception as error: blockers.append(dict(dataset=key[0],K=key[1],seed=key[2],reason=str(error)))
    ready=dict(ready=not blockers,formal_conditions=40,blockers=blockers)
    (a.out/'readiness.json').write_text(json.dumps(ready,indent=2),encoding='utf-8')
    if blockers:
        print(json.dumps(dict(status='blocked_missing_formal_assets',blocked_conditions=len(blockers),
            readiness=str(a.out/'readiness.json'),training_or_test_started=False),indent=2))
        return 2
    commands=[]
    for d,k,s in expected:
        dest=a.out/d/f'K{k}'/f'seed{s}'
        common=[sys.executable,'-m','cbrc']
        condition=['--assets',str(a.assets),'--dataset',d,'--K',str(k),'--seed',str(s)]
        commands.extend([
            common+['export',*condition,'--split','valid','--out',str(dest/'valid'),'--device',a.device],
            common+['fit','--export',str(dest/'valid'),'--out',str(dest/'fit')],
            common+['export',*condition,'--split','test','--out',str(dest/'test'),'--device',a.device,
                    '--selection',str(dest/'fit/selection.json')],
            common+['evaluate','--export',str(dest/'test'),'--selection',str(dest/'fit/selection.json'),
                    '--out',str(dest/'evaluation')]])
    results=[str(a.out/d/f'K{k}'/f'seed{s}'/'evaluation/evaluation.json') for d,k,s in expected]
    commands.append([sys.executable,'-m','cbrc','summarize','--results',*results,'--out',str(a.out/'summary')])
    (a.out/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    if a.execute:
        for command in commands: subprocess.run(command,check=True,cwd=Path(__file__).resolve().parents[1])
    print(json.dumps(dict(status='executed' if a.execute else 'ready_dry_run',commands=len(commands)),indent=2))
    return 0


if __name__=='__main__': sys.exit(main())
