"""Synthetic end-to-end correctness run. Never enters the paper's result tables."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

from .adapter import sha256,load_frozen
from .export import export_split
from .pipeline import fit,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args(); out=args.out.resolve()
    if out.exists(): raise FileExistsError('use a fresh smoke directory')
    out.mkdir(parents=True)
    sys.path.insert(0,str(args.code_root.resolve()))
    data=importlib.import_module('src.data');llm=importlib.import_module('src.llm_cache')
    modelmod=importlib.import_module('src.model')
    dataset=out/'toy_data';dataset.mkdir()
    splits={}
    for split,times in [('train',range(5)),('valid',range(5,15)),('test',range(15,18))]:
        rows=[((t+j)%12,j%2,((t+j)%12+j%2+1)%12,t) for t in times for j in range(4)]
        splits[split]=rows
        (dataset/f'{split}.txt').write_text(''.join(' '.join(map(str,q))+'\n' for q in rows),encoding='utf-8')
    kg=data.load_temporal_kg(str(dataset))
    torch.manual_seed(42)
    cfg=dict(shot=5,seed=42,dim=16,channels=4,history_len=4,dropout=0.,disable_support_gate=True,
        alterego_candidate_k=8,alterego_tournament_rank=4,llm_max_candidates=10,llm_max_delta=.35,
        fusion_mode='probability_mixture',llm_residual_control='hard')
    model=modelmod.NineFuseTKG(kg.num_entities,kg.num_relations*2,kg.num_times,
        dim=16,channels=4,history_len=4,dropout=0.,use_support_gate=False,
        alterego_candidate_k=8,alterego_tournament_rank=4,llm_mode='rationale',llm_residual_control='hard')
    checkpoint=out/'synthetic_untrained_graph.pt'
    torch.save(dict(model=model.state_dict(),args=cfg),checkpoint)
    before=sha256(checkpoint)
    cache_paths={}
    for split in ('valid','test'):
        cp=out/f'synthetic_{split}.jsonl'; cache_paths[split]=str(cp)
        with cp.open('w',encoding='utf-8') as f:
            seen=set()
            for s,r,_,t in data.add_inverse(splits[split],kg.num_relations):
                if (s,r,t) in seen: continue
                seen.add((s,r,t))
                q=dict(dataset_fingerprint=llm.dataset_files_fingerprint(str(dataset)),split=split,
                    direction='object' if r<kg.num_relations else 'subject',known_entity_id=s,
                    oriented_relation_id=r,timestamp=t,shot=5,seed=42,
                    history_protocol='standard_rolling_history',support_digest='synthetic',
                    history_digest='synthetic',prompt_template_version='synthetic-no-provider')
                # Proposals use only public s/r/t, never the hidden answer column.
                candidates=[dict(mapped_entity_id=v,confidence=.6,mapping_score=1.,
                    template_agreement=.5,temporal_score=.5) for v in ((s+1)%12,(s+2)%12,(s-1)%12)]
                f.write(json.dumps(dict(schema_version=1,query=q,query_key=llm.target_blind_query_key(q),
                    prompt_hash=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest(),candidates=candidates))+'\n')
    spec=dict(dataset='TOY',K=5,seed=42,checkpoint=str(checkpoint),data_dir=str(dataset),
              valid_cache=cache_paths['valid'],test_cache=cache_paths['test'])
    valid=export_split(spec,args.code_root,'valid',out/'valid',args.device,batch_size=4,smoke=True)
    selection=fit(out/'valid',out/'fit',epochs=6)
    test=export_split(spec,args.code_root,'test',out/'test',args.device,batch_size=4,smoke=True,
                      selection=out/'fit/selection.json')
    result=evaluate(out/'test',out/'fit/selection.json',out/'evaluation')
    # Direct inference check: attached semantic candidates cannot change the
    # graph anchor or tournament bank when its runtime semantic mode is off.
    train=importlib.import_module('src.train')
    frozen,_=load_frozen(args.code_root,checkpoint,kg,torch.device(args.device))
    assert not any(p.requires_grad for p in frozen.parameters())
    q=kg.valid[0]; public=[(q[0],q[1],0,q[3])]
    hist=data.HistoryIndex(kg.train_aug,kg.num_entities,kg.num_relations*2,4)
    cache=llm.LLMEvidenceCache(cache_paths['valid'])
    f=train.move_features(cache.augment_features(public,hist.build(public)),torch.device(args.device))
    supports=train.support_rows_tensor(train.causal_support_rows(train.group_by_relation(kg.train_aug),public,5),torch.device(args.device))
    with torch.inference_mode():
        scores0,aux0=frozen(data.tensorize(public,torch.device(args.device)),supports,f)
        changed={k:v.clone() for k,v in f.items()}
        changed['llm_candidate_ids']=(changed['llm_candidate_ids']+3)%kg.num_entities
        changed['llm_confidence']=1-changed['llm_confidence']
        altered=[(q[0],q[1],(q[2]+5)%kg.num_entities,q[3])]
        scores1,aux1=frozen(data.tensorize(altered,torch.device(args.device)),supports,changed)
        assert torch.equal(scores0,scores1)
        assert torch.equal(aux0['alterego_candidate_ids'],aux1['alterego_candidate_ids'])
    assert sha256(checkpoint)==before
    assert valid['queries']==80 and test['queries']==24
    assert all(m['queries']==24 for m in result['methods'].values())
    for metric in result['methods'].values():
        assert abs(metric['positive_rr_contribution']+metric['negative_rr_contribution']-
                   (metric['mrr']-result['methods']['graph']['mrr']))<1e-10
    # The sparse head never has access to the backbone or changes its parameters.
    manifest=dict(status='PASS',synthetic=True,untrained_graph=True,not_paper_evidence=True,
        graph_checkpoint_unchanged=True,validation_queries=80,test_queries=24,methods=list(result['methods']),
        parameter_count=193,frozen_state_dict_strict_loading=True)
    manifest.update(graph_bank_invariant_to_semantic_candidates=True,graph_scores_invariant_to_target=True,
                    all_backbone_parameters_frozen=True)
    (out/'smoke_verification.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
