"""Durable target-blind local Qwen3 evidence, using the recorded prompt and mapper.

No model call occurs on import or during training. This is an explicit generation
command; regenerated responses are new evidence, not the archived paper cache.
"""
from _common import ROOT,CODE,sha,read,write,verify_sources,lock
from pathlib import Path
import argparse
import bisect
import hashlib
import json
import os
import shutil
import sqlite3
import time
import urllib.request
from collections import defaultdict
from experiment import check_data
from src.data import load_temporal_kg,HistoryIndex,add_inverse
from src.train import choose_causal_support,group_by_relation
from src.stlp import TargetBlindQuery,build_stlp_prompt,build_query_metadata,EntityMapper,parse_and_map_response
from src.llm_cache import dataset_files_fingerprint,target_blind_query_key,LLMEvidenceCache,cache_coverage

MODEL='qwen3:latest'
DIGEST='500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
OPTIONS=dict(temperature=0.,top_p=1.,seed=42,num_ctx=4096,num_predict=1024)

def id_map(path,size):
    """YAGO: name and ID only; never expose later lifecycle-date columns."""
    mapping={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields=line.replace('\\t','\t').split('\t')
        if len(fields)<2: fields=line.split()
        if len(fields)<2: raise ValueError('Malformed YAGO mapping')
        mapping[fields[0]]=int(fields[1])
    if set(mapping.values())!=set(range(size)):raise ValueError('ID vocabulary mismatch')
    names=['']*size
    for name,index in mapping.items():names[index]=name
    return mapping,names

def api(endpoint,route,payload=None):
    req=urllib.request.Request(endpoint.rstrip('/')+route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)

def parse_attempts(request,mapper,support_names):
    """Only response-format errors become abstentions; transport errors propagate."""
    attempts=[]
    for _ in range(3):
        raw=request(); item=dict(raw=raw);attempts.append(item)
        try:
            content=raw.get('message',{}).get('content')
            if not isinstance(content,str):raise ValueError('message.content must be text')
            candidates,diag=parse_and_map_response(content,mapper,support_names,max_candidates=10)
        except (ValueError,TypeError,AttributeError) as exc:
            item['parse_error']=str(exc)
        else:return candidates,diag,attempts,False
    return [],dict(generation_failed=True,attempts=3),attempts,True

def generate(a):
    verify_sources();check_data('YAGO',a.data_dir)
    kg=load_temporal_kg(str(a.data_dir)); fingerprint=dataset_files_fingerprint(str(a.data_dir))
    mappings,names=id_map(a.data_dir/'entity2id.txt',kg.num_entities)
    _,rnames=id_map(a.data_dir/'relation2id.txt',kg.num_relations);mapper=EntityMapper(mappings)
    subset=read(ROOT/'configs/yago_transfer_subset.json') if a.scope=='transfer' else None
    if subset and (a.split!='test' or a.K!=5):raise ValueError('Published transfer scope is test, K=5 only')
    if subset and subset['dataset_fingerprint']!=fingerprint:raise ValueError('Subset fingerprint mismatch')
    selected_keys=None
    if subset:
        selected=[kg.test[i] for i in subset['source_line_indices_zero_based']]
        selected_keys={(s,r,t) for s,r,o,t in add_inverse(selected,kg.num_relations)}
    version='stlp-local-qwen3-yago-fixed-query-transfer-v1' if subset else 'stlp-local-qwen3-8b-yago-v1'
    config=dict(dataset_fingerprint=fingerprint,K=a.K,split=a.split,scope=a.scope,
        endpoint=a.endpoint,model=MODEL,digest=DIGEST,options=OPTIONS,thinking=False,
        generator_sha256=sha(__file__),prompt_template_version=version,
        response_policy='three-format-failures-abstain; transport-errors-stop')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out=a.out.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with lock(Path(str(out)+'.lockdir')):
        if out.exists():
            marker=Path(str(out)+'.complete.json')
            if not marker.exists() or read(marker)['signature']!=signature or read(marker)['sha256']!=sha(out):
                raise ValueError('Existing cache is incomplete or belongs to another configuration')
            print('Already complete:',out);return
        models=[m for m in api(a.endpoint,'/api/tags')['models'] if m['name']==MODEL]
        if len(models)!=1 or models[0]['digest']!=DIGEST:raise ValueError('Pinned Qwen3 digest absent or changed')
        db=sqlite3.connect(str(out)+'.sqlite')
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS config(signature TEXT PRIMARY KEY,value TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS records(key TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,payload TEXT,raw TEXT)')
        saved=db.execute('SELECT signature FROM config').fetchall()
        if saved and saved!=[(signature,)]:raise ValueError('Cache resumption configuration changed')
        if not saved:db.execute('INSERT INTO config VALUES (?,?)',(signature,json.dumps(config)));db.commit()
        done={r[0] for r in db.execute('SELECT key FROM records')}
        initial=kg.train_aug+(kg.valid_aug if a.split=='test' else [])
        history=HistoryIndex(initial,kg.num_entities,kg.num_relations*2,history_len=16)
        grouped=group_by_relation(initial);snapshots=defaultdict(list)
        for fact in (kg.test if a.split=='test' else kg.valid):snapshots[fact[3]].append(fact)
        seen=set();ordinal=0
        for t in sorted(snapshots):
            oriented=add_inverse(snapshots[t],kg.num_relations)
            for s,r,_hidden,_ in oriented:
                loc=(s,r,t)
                if loc in seen or (selected_keys is not None and loc not in selected_keys):continue
                seen.add(loc);n=ordinal;ordinal+=1
                public=TargetBlindQuery(a.split,'tail' if r<kg.num_relations else 'head',s,r,t)
                support=choose_causal_support(grouped,(s,r,0,t),a.K)
                events=history.events_by_subject[s];i=bisect.bisect_left(events,(t,-1,-1))
                recent=[(s,rel,obj,tm) for tm,rel,obj in events[max(0,i-10):i]]
                if any(f[3]>=t for f in support+recent):raise ValueError('History barrier violated')
                meta=build_query_metadata(public,a.K,42,'standard_rolling_history',support,recent,fingerprint,version)
                key=target_blind_query_key(meta)
                if key in done:continue
                if shutil.disk_usage(out.parent).free<4*1024**3:raise RuntimeError('Disk below 4 GiB; cache transactions retained')
                prompt=build_stlp_prompt(public,support,recent,names,rnames,kg.num_relations,max_candidates=10).replace('ICEWS entity name','YAGO entity name')
                payload=dict(model=MODEL,stream=False,think=False,format='json',keep_alive='5m',options=OPTIONS,
                    messages=[dict(role='system',content='Return valid JSON only from the supplied strictly earlier context. /no_think'),dict(role='user',content=prompt)])
                start=time.perf_counter()
                candidates,diag,attempts,abstain=parse_attempts(lambda:api(a.endpoint,'/api/chat',payload),mapper,[names[x[2]] for x in support])
                record=dict(schema_version=1,query=meta,query_key=key,prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                    provider='local_ollama',model=MODEL,model_digest=DIGEST,response_id='local:'+key,
                    candidates=candidates,diagnostics=diag,abstain=abstain,latency_ms=1000*(time.perf_counter()-start),generated_unix=time.time())
                db.execute('INSERT INTO records VALUES (?,?,?,?)',(key,n,json.dumps(record,ensure_ascii=False),json.dumps(attempts,ensure_ascii=False)))
                db.commit();done.add(key)
                write(str(out)+'.progress.json',dict(completed=len(done),unique_seen=ordinal,abstained_last=abstain))
            # Full observed feedback after the whole timestamp, including unscored facts.
            for f in oriented:grouped.setdefault(f[1],[]).append(f)
            for r in {f[1] for f in oriented}:grouped[r].sort(key=lambda f:(f[3],f[0],f[2]))
            history.add_facts(oriented)
        if len(done)!=ordinal:raise ValueError('Cache record count mismatch')
        tmp=Path(str(out)+'.tmp');failed=0
        with tmp.open('w',encoding='utf-8') as f:
            for payload, in db.execute('SELECT payload FROM records ORDER BY ordinal'):
                failed+=bool(json.loads(payload).get('abstain'));f.write(payload+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,out);db.close()
        write(str(out)+'.meta.json',dict(schema_version=2,shot=a.K,split=a.split,dataset_fingerprint=fingerprint,
            history_protocol='standard_rolling_history',query_key_excludes_target=True,prompt_template_version=version,
            provider_provenance=dict(provider='local_ollama',model=MODEL,digest=DIGEST,options=OPTIONS,thinking=False),
            generation_audit=dict(records=ordinal,signature=signature,subset=bool(subset),generator_sha256=sha(__file__)),cache_sha256=sha(out)))
        cache=LLMEvidenceCache(str(out),expected_shot=a.K,expected_split=a.split,expected_dataset_fingerprint=fingerprint,
            expected_history_protocol='standard_rolling_history',require_generation_metadata=True)
        target=add_inverse(selected if subset else (kg.test if a.split=='test' else kg.valid),kg.num_relations)
        coverage=cache_coverage(cache,target)
        if coverage['cache_hit_rate']!=1:raise ValueError('Incomplete cache coverage')
        write(str(out)+'.complete.json',dict(complete=True,sha256=sha(out),signature=signature,records=ordinal,
            abstentions=failed,coverage=coverage,scope=a.scope,prediction_instances=len(target)))
        print('Completed:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--K',type=int,choices=[5,10],required=True)
    p.add_argument('--split',choices=['valid','test'],required=True);p.add_argument('--scope',choices=['full','transfer'],default='full')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--endpoint',default='http://127.0.0.1:11434')
    generate(p.parse_args())

if __name__=='__main__':main()
