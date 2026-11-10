"""Sequential, resumable seed/budget launcher; never starts evidence generation."""
from _common import ROOT,lock
from pathlib import Path
import argparse
import json
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['train','calibrate','paired'],required=True)
    p.add_argument('--dataset',choices=['ICEWS14','YAGO'],required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--shots',type=int,nargs='+',choices=[5,10],default=[5,10])
    p.add_argument('--seeds',type=int,nargs='+',choices=range(42,52),default=list(range(42,52)))
    p.add_argument('--device',default='cuda:0');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.stage!='train' and a.cache_dir is None:p.error('--cache-dir is required for evaluation/calibration')
    if len(set(a.shots))!=len(a.shots) or len(set(a.seeds))!=len(a.seeds):p.error('Duplicate conditions')
    commands=[]
    for k in a.shots:
        for seed in a.seeds:
            dest=a.run_root.resolve()/a.dataset/f'K{k}'/f'seed{seed}'
            command=[sys.executable,str(ROOT/'scripts/experiment.py'),a.stage,
                '--dataset',a.dataset,'--data-dir',str(a.data_dir.resolve()),'--K',str(k),
                '--seed',str(seed),'--device',a.device]
            if a.stage=='train':command+=['--out',str(dest)]
            else:
                command+=['--checkpoint',str(dest/'graph/best.pt')]
                if a.stage=='calibrate':
                    command+=['--valid-cache',str(a.cache_dir.resolve()/f'{a.dataset.lower()}_valid_k{k}.jsonl'),
                        '--test-cache',str(a.cache_dir.resolve()/f'{a.dataset.lower()}_test_k{k}.jsonl'),
                        '--out',str(dest/'calibration')]
                else:
                    if (dest/'paired/evaluation.json').exists():continue
                    command+=['--cache',str(a.cache_dir.resolve()/f'{a.dataset.lower()}_test_k{k}.jsonl'),
                        '--out',str(dest/'paired')]
            commands.append(command)
    if a.dry_run:print(json.dumps(commands,indent=2));return
    with lock(a.run_root.resolve()/'matrix_lock'):
        for command in commands:subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
