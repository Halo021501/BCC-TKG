"""Read-only adapter to the paper's frozen source; never changes its state dict."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def implementation_sha256():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def load_frozen(code_root, checkpoint_path, kg, device):
    code_root = Path(code_root).resolve()
    if 'src.model' in sys.modules:
        loaded = Path(sys.modules['src.model'].__file__).resolve()
        if code_root not in loaded.parents:
            raise RuntimeError('another src.model is imported; use a fresh process')
    sys.path.insert(0, str(code_root))
    module = importlib.import_module('src.model')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = checkpoint['args']
    if not isinstance(cfg, dict):
        raise ValueError('checkpoint args must be a safe plain mapping')
    if not cfg.get('disable_support_gate', False) or cfg.get('disable_support', False):
        raise ValueError('requires an ungated checkpoint retaining support, not a runtime override')
    kwargs = {name: not cfg.get('disable_'+short, False) for name,short in [
        ('use_copy','copy'),('use_rel_copy','rel_copy'),('use_rule','rule'),('use_freq','freq'),
        ('use_history','history'),('use_support','support'),('use_router','router'),
        ('use_temporal_calibration','temporal_calibration'),('use_snapshot_backbone','snapshot_backbone'),
        ('use_candidate_rerank','candidate_rerank'),('use_alterego_tournament','alterego_tournament')]}
    kwargs.update(use_support_gate=False, use_geo=cfg.get('enable_geo',False) and not cfg.get('disable_geo',False))
    for key,default in dict(dim=256, history_len=10, channels=64, dropout=.2,
        alterego_candidate_k=96, alterego_tournament_rank=32, alterego_max_delta=.5,
        llm_max_candidates=10, llm_max_delta=.35, llm_score_scale=1., llm_disable_confidence=False,
        expert_dropout=.05, max_residual_gate=.8, max_expert_mass=.65,
        fusion_mode='probability_mixture', llm_residual_control='hard').items():
        kwargs[key]=cfg.get(key,default)
    kwargs['llm_mode']='rationale'  # instantiate all legacy parameters; runtime off below
    model=module.NineFuseTKG(kg.num_entities,kg.num_relations*2,kg.num_times,**kwargs)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(device).eval().requires_grad_(False)
    model.set_llm_runtime_mode('off')
    return model,cfg


def verified_assets(spec, code_root, split, smoke=False):
    if split not in ('valid','test'):
        raise ValueError('only validation calibration and final test are supported')
    paths={k:Path(spec[k]).resolve() for k in ('checkpoint','data_dir','valid_cache','test_cache')}
    needed=('checkpoint','data_dir',split+'_cache')
    for key in needed:
        if not str(spec[key]).strip() or not (paths[key].is_dir() if key=='data_dir' else paths[key].is_file()):
            raise FileNotFoundError(f'missing {key}: {paths[key]}')
    identities={'checkpoint':sha256(paths['checkpoint']),split+'_cache':sha256(paths[split+'_cache'])}
    if not smoke:
        if spec.get('status') not in ('verified_formal','verified_new_experiment'):
            raise ValueError('formal runs require status=verified_formal with recovered original asset provenance')
        if spec.get('status')=='verified_new_experiment' and not spec.get('experiment_id'):
            raise ValueError('newly trained experiments require a separate experiment identifier')
        for key,value in identities.items():
            if spec.get('sha256',{}).get(key) != value:
                raise ValueError(f'missing or mismatched registered hash: {key}')
        manifest=Path(code_root)/'provenance/source_manifest.original.json'
        original=json.loads(manifest.read_text(encoding='utf-8'))
        for entry in original['files']:
            relative=entry['path'].split('/code/')[-1]
            p=Path(code_root)/relative
            if not p.is_file() or sha256(p)!=entry['sha256']:
                raise ValueError(f'frozen source mismatch: {relative}')
        for name,expected in spec['dataset_sha256'].items():
            if sha256(paths['data_dir']/name)!=expected:
                raise ValueError(f'dataset mismatch: {name}')
        if not spec.get('recovery_provenance'):
            raise ValueError('document the origin of recovered weights/caches; hash alone is not provenance')
    return paths,identities
