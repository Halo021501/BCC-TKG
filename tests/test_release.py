import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT),str(ROOT/'frozen_graph_code')]
from _common import verify_sources,lock
from generate_local_cache import parse_attempts,id_map
from src.stlp import EntityMapper

class ReleaseTests(unittest.TestCase):
    def test_frozen_sources_match_manifest(self):verify_sources()
    def test_three_format_failures_abstain_and_keep_raw(self):
        responses=[]
        def request():
            responses.append(1);return {'message':{'content':'invalid json'}}
        candidates,diag,attempts,abstain=parse_attempts(request,EntityMapper({'entity':0}),[])
        self.assertTrue(abstain);self.assertEqual(candidates,[]);self.assertEqual(len(responses),3)
        self.assertTrue(all('raw' in a and 'parse_error' in a for a in attempts))
    def test_transport_error_does_not_become_abstention(self):
        def request():raise ConnectionError('test-only network failure')
        with self.assertRaises(ConnectionError):parse_attempts(request,EntityMapper({'entity':0}),[])
    def test_yago_lifecycle_dates_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'mapping.txt';p.write_text('Entity_A\t0\t1900\t2000\nEntity_B\t1\t1901\t2001\n')
            m,names=id_map(p,2)
            self.assertEqual(names,['Entity_A','Entity_B']);self.assertEqual(m['Entity_A'],0)
    def test_execution_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            with lock(directory):pass
            with lock(directory):pass

if __name__=='__main__':unittest.main()
