import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='CBRC: freeze graph ranking, fit bounded candidate corrections, evaluate once')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit'); a.add_argument('--workspace',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('export'); a.add_argument('--assets',type=Path,required=True); a.add_argument('--dataset',choices=['ICEWS14','YAGO','TOY'],required=True)
    a.add_argument('--K',type=int,choices=[5,10],required=True); a.add_argument('--seed',type=int,required=True)
    a.add_argument('--split',choices=['valid','test'],required=True); a.add_argument('--out',type=Path,required=True)
    a.add_argument('--device',default='cpu'); a.add_argument('--batch-size',type=int,default=64)
    a.add_argument('--selection',type=Path); a.add_argument('--smoke',action='store_true'); a.add_argument('--limit',type=int,default=0)
    a=sub.add_parser('fit'); a.add_argument('--export',dest='export_path',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    a.add_argument('--epochs',type=int,default=30)
    a=sub.add_parser('evaluate'); a.add_argument('--export',dest='export_path',type=Path,required=True)
    a.add_argument('--selection',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('summarize'); a.add_argument('--results',type=Path,nargs='*',default=[]); a.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.command=='audit':
        from .audit import inventory
        inventory(args.workspace,args.out); return
    if args.command=='export':
        from .export import export_split
        assets=json.loads(args.assets.read_text(encoding='utf-8'))
        matches=[c for c in assets['conditions'] if (c['dataset'],c['K'],c['seed'])==(args.dataset,args.K,args.seed)]
        if len(matches)!=1: p.error('exactly one matching asset condition required')
        result=export_split(matches[0],assets['code_root'],args.split,args.out,args.device,args.batch_size,args.smoke,args.selection,args.limit)
        result={k:v for k,v in result.items() if k!='shards'}
    elif args.command=='fit':
        from .pipeline import fit
        result=fit(args.export_path,args.out,args.epochs)
        result={k:v for k,v in result.items() if k!='variants'}
    elif args.command=='evaluate':
        from .pipeline import evaluate
        result=evaluate(args.export_path,args.selection,args.out)
    else:
        from .statistics import summarize
        result=summarize(args.results,args.out)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
