"""Repository-relative setup and atomic experiment I/O."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'frozen_graph_code'
sys.path[:0] = [str(ROOT), str(CODE), str(ROOT/'runtime')]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def verify_sources():
    for entry in read(CODE/'provenance/source_manifest.original.json')['files']:
        rel=entry['path'].split('/code/')[-1]
        if sha(CODE/rel)!=entry['sha256']: raise ValueError('Frozen source changed: '+rel)

@contextlib.contextmanager
def lock(directory):
    """An OS-released lock; an interrupted run does not leave a stale PID lock."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.execution.lock').open('a+b') as f:
        f.seek(0)
        if not f.read(1): f.write(b'1'); f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f.fileno(),fcntl.LOCK_UN)
