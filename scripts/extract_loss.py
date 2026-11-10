"""Extract actual frozen-trainer losses from a run-matrix directory."""
from _common import ROOT
from pathlib import Path
import argparse
import csv
import re

PATTERN=re.compile(r'^(warmup_epoch|epoch)=(\d+) loss=([0-9.eE+-]+)(?: nll=([0-9.eE+-]+)| epi_nll=([0-9.eE+-]+) sup_nll=([0-9.eE+-]+))')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();records=[]
    for logfile in sorted(a.run_root.glob('*/K*/seed*/train.log')):
        dataset,k,seed=logfile.relative_to(a.run_root).parts[:3];seen={}
        for line in logfile.read_text(encoding='utf-8',errors='replace').splitlines():
            m=PATTERN.match(line)
            if not m:continue
            phase='warmup' if m[1]=='warmup_epoch' else 'joint';epoch=int(m[2])
            row=dict(study=dataset,K=int(k[1:]),seed=int(seed[4:]),phase=phase,epoch=epoch,
                     loss=float(m[3]),episodic_nll=m[5] or '',supervised_nll=m[6] or '')
            key=(phase,epoch)
            if key in seen and seen[key]!=row:raise ValueError(f'Conflicting repeated epoch in {logfile}: {key}')
            seen[key]=row
        records.extend(seen.values())
    if not records:raise ValueError('No recorded epoch losses found')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('Recorded loss rows:',len(records))

if __name__=='__main__':main()
