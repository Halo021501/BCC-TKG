"""Reproduce Figure 5 from recorded training losses; no text/image generation."""
from _common import ROOT
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'results_reference/training_loss.csv')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with a.data.open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(7.08,2.60),layout='constrained')
    for ax,(dataset,k),letter in zip(axes,[('ICEWS14',5),('ICEWS14',10),('YAGO',5)],'abc'):
        lines=[]
        for seed in range(42,52):
            rr={int(r['epoch']):float(r['loss']) for r in records if r['study']==dataset and int(r['K'])==k and int(r['seed'])==seed and r['phase']=='joint'}
            if set(rr)!=set(range(1,41)):raise ValueError('Missing or duplicate training epochs')
            lines.append([rr[e] for e in range(1,41)])
        x=np.arange(1,41);y=np.array(lines);color='#286A9A'
        for r in y:ax.plot(x,r,color=color,alpha=.22,lw=.65)
        ax.fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=color,alpha=.14,lw=0)
        ax.plot(x,y.mean(0),color=color,lw=1.8)
        ax.set(xlabel='Joint-training epoch',title=f'{dataset}, K = {k}',xlim=(1,40))
        ax.text(-.12,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=10,va='top')
    axes[0].set_ylabel('Recorded joint-training loss')
    for ext in ('pdf','svg','png'):fig.savefig(a.out/f'fig5_training_loss.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
