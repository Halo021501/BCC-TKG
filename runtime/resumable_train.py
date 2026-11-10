"""Instrument the pinned trainer with durable resume hooks; frozen sources stay intact.

Only loop cursors, state persistence, and atomic best-model writes are added.
The exact instrumented source and its digest are saved for review.
"""
from pathlib import Path
import hashlib,importlib,json,os,random,sys,time,traceback
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CODE=ROOT/'frozen_graph_code'
EXPECTED='d8db52ca6a876c5cc67ef2087c58b38d79a57acdf9dab4c15d49767977f6f085'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def atomic_torch(value,path):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

class ResumeController:
 def __init__(self):self.loaded=None;self.keep_previous=False;self.last_saved=0.;self.interval=float(os.environ.get('CBRC_CHECKPOINT_SECONDS','120'))
 def bind(self,args,model,optimizer,meta):
  self.args=args;self.model=model;self.optimizer=optimizer;self.path=Path(args.output_dir)/'last.pt';self.prev=self.path.with_name('previous.pt')
  self.signature=hashlib.sha256(json.dumps(dict(args=vars(args),adapter_sha256=sha(__file__),trainer_sha256=EXPECTED),sort_keys=True).encode()).hexdigest()
  errors=[]
  for path in (self.path,self.prev):
   if not path.exists():continue
   try:
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['signature']!=self.signature:raise ValueError('Resume configuration/source signature mismatch')
    self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer'])
    self.restore_global(state);self.loaded=state;self.keep_previous=(path==self.prev)
    print(f"RESUMED checkpoint={path} phase={state['phase']} epoch={state['epoch']} next_step={state['next_step']}",flush=True)
    break
   except Exception as e:
    errors.append(f'{path}: {e}')
    if 'signature mismatch' in str(e):raise
  if errors and self.loaded is None:raise RuntimeError('No readable checkpoint; refusing silent restart: '+'; '.join(errors))
  meta['resume_execution']=dict(adapter=str(Path(__file__).resolve()),adapter_sha256=sha(__file__),original_train_sha256=EXPECTED,
    signature=self.signature,checkpoint_interval_seconds=self.interval,state='model, AdamW, global and sampler RNG, exact next batch, shuffled warmup facts, losses and best validation',
    restored_checkpoint_phase=self.loaded['phase'] if self.loaded else None)
  atomic_json(Path(args.output_dir)/'run_meta.json',meta)
 def restore_global(self,state):
  random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
  if state['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
 def save(self,phase,epoch,next_step,extra,force=False):
  if not force and time.monotonic()-self.last_saved<self.interval:return
  value=dict(schema_version=1,signature=self.signature,phase=phase,epoch=epoch,next_step=next_step,
   model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,**extra)
  temp=self.path.with_suffix('.pt.tmp')
  with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  if self.path.exists() and not self.keep_previous:os.replace(self.path,self.prev)
  os.replace(temp,self.path);self.keep_previous=False;self.last_saved=time.monotonic()
  atomic_json(self.path.parent/'resume_progress.json',dict(phase=phase,epoch=epoch,next_step=next_step,checkpoint=str(self.path),saved_unix=time.time(),pid=os.getpid()))
  print(f'CHECKPOINT phase={phase} epoch={epoch} next_step={next_step}',flush=True)
  if os.environ.get('CBRC_TEST_STOP_PHASE')==phase and next_step==int(os.environ.get('CBRC_TEST_STOP_STEP','1')):
   print('TEST_ONLY forced process termination after durable checkpoint',flush=True);os._exit(86)
 def skip_warmup(self):return self.loaded is not None and self.loaded['phase']!='warmup'
 def warm_start(self):return self.loaded['epoch'] if self.loaded and self.loaded['phase']=='warmup' else 1
 def warm_epoch(self,epoch,rng,facts):
  if self.loaded and self.loaded['phase']=='warmup' and epoch==self.loaded['epoch']:
   s=self.loaded;rng.setstate(s['warm_rng']);facts[:]=s['facts']
   return s['next_step'],s['total_loss'],s['total_nll'],s['seen']
  rng.shuffle(facts);return 0,0.,0.,0
 def warm_step(self,epoch,step,rng,facts,loss,nll,seen,force):
  self.save('warmup',epoch,step,dict(warm_rng=rng.getstate(),facts=facts,total_loss=loss,total_nll=nll,seen=seen),force)
 def joint_start(self,rng,supervised_rng,sampler,supervised_sampler):
  if self.loaded and self.loaded['phase']=='joint':
   s=self.loaded
   for key,obj in [('joint_rng',rng),('supervised_rng',supervised_rng),('sampler_rng',sampler.rng),('supervised_sampler_rng',supervised_sampler.rng)]:obj.setstate(s[key])
   self.restore_global(s)
   return s['epoch'],s['best_valid'],s['last_metrics']
  return 1,-1.,{}
 def joint_epoch(self,epoch):
  if self.loaded and self.loaded['phase']=='joint' and epoch==self.loaded['epoch']:
   s=self.loaded;return s['next_step'],s['total_loss'],s['total_epi_nll'],s['total_sup_nll']
  return 0,0.,0.,0.
 def joint_step(self,epoch,step,rng,supervised_rng,sampler,supervised_sampler,loss,epi,sup,best,metrics,force=False):
  self.save('joint',epoch,step,dict(joint_rng=rng.getstate(),supervised_rng=supervised_rng.getstate(),sampler_rng=sampler.rng.getstate(),
   supervised_sampler_rng=supervised_sampler.rng.getstate(),total_loss=loss,total_epi_nll=epi,total_sup_nll=sup,best_valid=best,last_metrics=metrics),force)
 def atomic_model_save(self,value,path):atomic_torch(value,path)

def instrument(source):
 def replace(old,new):
  nonlocal source
  if source.count(old)!=1:raise RuntimeError('Pinned trainer anchor mismatch: '+old[:90])
  source=source.replace(old,new,1)
 replace('    if args.warmup_epochs <= 0:\n        return','    if args.warmup_epochs <= 0 or _RESUME.skip_warmup():\n        return')
 replace('    for epoch in range(1, args.warmup_epochs + 1):\n        model.train()\n        total_loss = 0.0\n        total_nll = 0.0\n        rng.shuffle(facts)\n        seen = 0\n        for batch_index in range(batches_per_epoch):',
         '    for epoch in range(_RESUME.warm_start(), args.warmup_epochs + 1):\n        model.train()\n        resume_step, total_loss, total_nll, seen = _RESUME.warm_epoch(epoch, rng, facts)\n        for batch_index in range(resume_step, batches_per_epoch):')
 replace('            seen += len(batch)\n','            seen += len(batch)\n            _RESUME.warm_step(epoch, batch_index+1, rng, facts, total_loss, total_nll, seen, batch_index+1==batches_per_epoch)\n')
 replace('    model.set_alterego_runtime_enabled(False)\n    model.set_llm_runtime_mode("off")\n    run_supervised_warmup(',
         '    _RESUME.bind(args, model, optimizer, meta)\n    model.set_alterego_runtime_enabled(False)\n    model.set_llm_runtime_mode("off")\n    run_supervised_warmup(')
 replace('    for epoch in range(1, args.epochs + 1):\n        model.train()\n        total_loss = 0.0\n        total_epi_nll = 0.0\n        total_sup_nll = 0.0\n        for _ in range(args.episodes_per_epoch):',
         '    resume_epoch, best_valid, last_metrics = _RESUME.joint_start(rng, supervised_rng, sampler, supervised_sampler)\n    for epoch in range(resume_epoch, args.epochs + 1):\n        model.train()\n        resume_step, total_loss, total_epi_nll, total_sup_nll = _RESUME.joint_epoch(epoch)\n        for episode_index in range(resume_step, args.episodes_per_epoch):')
 replace('            total_sup_nll += float(sup_nll.item())\n','            total_sup_nll += float(sup_nll.item())\n            _RESUME.joint_step(epoch, episode_index+1, rng, supervised_rng, sampler, supervised_sampler, total_loss, total_epi_nll, total_sup_nll, best_valid, last_metrics)\n')
 replace('    if os.path.exists(best_path):','        _RESUME.joint_step(epoch+1, 0, rng, supervised_rng, sampler, supervised_sampler, 0., 0., 0., best_valid, last_metrics, force=True)\n\n    if os.path.exists(best_path):')
 replace('                torch.save(\n','                _RESUME.atomic_model_save(\n')
 return source

def main():
 sys.path.insert(0,str(CODE));module=importlib.import_module('src.train');path=CODE/'src/train.py'
 if sha(path)!=EXPECTED:raise RuntimeError('Original trainer changed; do not instrument')
 source=instrument(path.read_text(encoding='utf-8'))
 module.__dict__['_RESUME']=ResumeController()
 exec(compile(source,str(HERE/'instrumented_train.py'),'exec'),module.__dict__)
 # A source snapshot makes the additional execution version explicit.
 snapshot=HERE/'instrumented_train.py'
 if snapshot.exists() and snapshot.read_text(encoding='utf-8')!=source:raise RuntimeError('Execution snapshot differs')
 if not snapshot.exists():snapshot.write_text(source,encoding='utf-8')
 module.main()

if __name__=='__main__':main()
