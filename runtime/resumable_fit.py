"""Original CBRC objective/selection with per-batch durable calibration state."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'CBRC'))
from cbrc import pipeline as p
from cbrc.core import CandidateCorrector,VARIANTS,SCALES,sparse_objective
from cbrc.adapter import sha256,implementation_sha256
from resumable_train import atomic_torch,atomic_json

def fit(export_path,out,epochs=30,patience=5,variants=VARIANTS):
 path,meta=p.read_export(export_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if meta['split']!='valid' or meta['cbrc_source_sha256']!=implementation_sha256():raise ValueError('Invalid frozen validation export')
 signature=hashlib.sha256((sha256(path/'export.json')+sha256(__file__)+str(epochs)+str(tuple(variants))).encode()).hexdigest()
 checkpoint=out/'fit_resume.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint.exists() else None
 if saved and saved['signature']!=signature:raise ValueError('Calibration resume signature changed')
 result=saved['completed'] if saved else {}
 if (out/'selection.json').exists():return json.loads((out/'selection.json').read_text())
 torch.set_num_threads(1);last_save=0.
 for variant in variants:
  if variant in result:
   if sha256(out/result[variant]['model_file'])!=result[variant]['sha256']:raise ValueError('Completed head changed')
   continue
  torch.manual_seed(meta['seed']);model=CandidateCorrector(variant)
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  best_mrr=p.assess(model,path,meta,'select',.35)['mrr'];best=copy.deepcopy(model.state_dict());best_epoch=0;stall=0;trace=[]
  start_epoch=1;start_batch=0;losses=[]
  if saved and saved.get('variant')==variant:
   model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
   best_mrr=saved['best_mrr'];best=saved['best'];best_epoch=saved['best_epoch'];stall=saved['stall'];trace=saved['trace']
   start_epoch=saved['epoch'];start_batch=saved['next_batch'];losses=saved['losses']
   print(f'RESUMED head={variant} epoch={start_epoch} next_batch={start_batch}',flush=True)
  def persist(epoch,next_batch,losses,force=False):
   nonlocal last_save
   if not force and time.monotonic()-last_save<120:return
   state=dict(signature=signature,completed=result,variant=variant,model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
    best_mrr=best_mrr,best=best,best_epoch=best_epoch,stall=stall,trace=trace,epoch=epoch,next_batch=next_batch,losses=losses)
   atomic_torch(state,checkpoint);last_save=time.monotonic()
   atomic_json(out/'fit_progress.json',dict(variant=variant,epoch=epoch,next_batch=next_batch,completed_variants=list(result),updated_unix=time.time()))
   if os.environ.get('CBRC_TEST_FIT_STOP')==variant and next_batch==1:os._exit(87)
  for epoch in range(start_epoch,epochs+1):
   if stall>=patience:break
   model.train()
   if epoch!=start_epoch:losses=[];start_batch=0
   for i,batch in enumerate(p.batches(path,meta,'fit',variant,seed=meta['seed']+epoch)):
    if i<start_batch:continue
    optimizer.zero_grad(set_to_none=True);delta=model(batch['z'],batch['mask'],.35)
    loss,parts=sparse_objective(batch['base'],batch['ids'],batch['mask'],delta,[r['truths'] for r in batch['records']],0. if variant=='no_kl' else 1.)
    if not torch.isfinite(loss):raise ValueError('Nonfinite calibration objective')
    loss.backward();optimizer.step();losses.append(float(loss));persist(epoch,i+1,losses)
   if not losses:raise ValueError('No calibration fit rows')
   model.eval();metric=p.assess(model,path,meta,'select',.35)['mrr'];trace.append(dict(epoch=epoch,loss=sum(losses)/len(losses),select_mrr=metric))
   if metric>best_mrr+1e-12:best_mrr=metric;best=copy.deepcopy(model.state_dict());best_epoch=epoch;stall=0
   else:stall+=1
   persist(epoch+1,0,[],force=True)
   if stall>=patience:break
  model.load_state_dict(best);model.eval();grid=[dict(scale=s,**p.assess(model,path,meta,'select',s)) for s in SCALES]
  chosen=max(grid,key=lambda m:(m['mrr'],-m['scale']));audit=p.assess(model,path,meta,'audit',chosen['scale'])
  model_file=f'{variant}.pt';atomic_torch(dict(model=model.state_dict(),variant=variant,scale=chosen['scale'],checkpoint_sha256=meta['checkpoint_sha256']),out/model_file)
  result[variant]=dict(model_file=model_file,sha256=sha256(out/model_file),scale=chosen['scale'],best_epoch=best_epoch,selection_grid=grid,development_audit=audit,trace=trace,parameter_count=193)
  atomic_torch(dict(signature=signature,completed=result,variant=None),checkpoint)
  saved=None
 selection=dict(schema_version=1,dataset=meta['dataset'],K=meta['K'],seed=meta['seed'],smoke=meta['smoke'],
  experiment_id=meta.get('experiment_id','original_block_b'),asset_origin=meta.get('asset_origin','verified_formal'),checkpoint_sha256=meta['checkpoint_sha256'],
  validation_cache_sha256=meta['cache_sha256'],dataset_fingerprint=meta['dataset_fingerprint'],source_model_sha256=meta['source_model_sha256'],cbrc_source_sha256=meta['cbrc_source_sha256'],
  valid_export_sha256=sha256(path/'export.json'),variants=result,development_audit_independent=False,supervision='validation calibration labels; graph frozen',
  training=dict(max_epochs=epochs,patience=5,lr=.001,weight_decay=.0001,kl_weight=1.,scales=list(SCALES)),resume_execution=dict(adapter_sha256=sha256(__file__),signature=signature))
 atomic_json(out/'selection.json',selection);return selection
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--export',dest='export_path',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
 args=a.parse_args();fit(args.export_path,args.out)
